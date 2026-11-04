"""STTran gọn (Cong et al., ICCV 2021) chạy trên đặc trưng đã trích bằng YOLO (extract_feats.py).

Mỗi token = một cặp (người, vật) trong một frame:
  [đặc trưng người, đặc trưng vật, đặc trưng union box, mặt nạ không gian 2x27x27 -> conv, embedding lớp vật, hình học]
  -> Spatial encoder : self-attention giữa các cặp CÙNG frame
  -> Temporal encoder: self-attention giữa các cặp của frame t và frame t-1..t-window+1 (cửa sổ trượt, mã hoá vị trí frame)
  -> 3 đầu: attention (softmax 3) | spatial (sigmoid 6) | contacting (sigmoid 17)
Khác bản gốc: (1) đặc trưng từ YOLO thay vì Faster R-CNN; (2) cửa sổ trượt áp dụng ở MỖI lớp temporal nên nhiều lớp
mở rộng tầm nhìn; (3) nhiều video ghép thành một chuỗi token với mặt nạ attention khối.
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.ops import box_iou

from agsgg.ag_dataset import _cxcywh
from agsgg.constants import N_ATT, N_SPA, N_CON, N_OBJ


# ----------------------------------------------------------------------------- dữ liệu
def make_target(vid, f):
    """Target kiểu agsgg (cxcywh chuẩn hoá theo kích thước nhãn) để dùng lại agsgg.evaluate."""
    w, h = f['size']
    pb = np.array(f['person']) if f['has_person'] else np.array([0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h], np.float32)
    boxes = torch.tensor([_cxcywh(b, w, h) for b in f['boxes']], dtype=torch.float32).reshape(-1, 4)
    return dict(person_box=torch.tensor(_cxcywh(pb, w, h)), size=(h, w), has_person=bool(f['has_person']),
                vid=vid, frame=f['frame'], obj_labels=torch.as_tensor(f['labels'], dtype=torch.long),
                obj_boxes=boxes, att=torch.as_tensor(f['att'], dtype=torch.long),
                spa=torch.as_tensor(f['spa'], dtype=torch.float32), con=torch.as_tensor(f['con'], dtype=torch.float32))


def gt_norm_boxes(f):
    w, h = f['size']
    return f['boxes'] / np.array([w, h, w, h], np.float32)


def assign_targets(rec, f, thr=0.5):
    """Gán nhãn quan hệ cho các vật YOLO phát hiện: GT cùng lớp, IoU >= thr (lớn nhất). -> att, spa, con, matched."""
    n = rec['n']
    att = np.full(n, -1, np.int64); spa = np.zeros((n, N_SPA), np.float32); con = np.zeros((n, N_CON), np.float32)
    matched = np.zeros(n, bool)
    if n == 0 or len(f['labels']) == 0:
        return att, spa, con, matched
    iou = box_iou(torch.from_numpy(rec['boxes']), torch.from_numpy(gt_norm_boxes(f))).numpy()
    same = rec['labels'][:, None].astype(np.int64) == f['labels'][None]
    iou = np.where(same, iou, -1)
    j = iou.argmax(1)
    ok = iou[np.arange(n), j] >= thr
    att[ok] = f['att'][j[ok]]; spa[ok] = f['spa'][j[ok]]; con[ok] = f['con'][j[ok]]
    matched[ok] = True
    return att, spa, con, matched


class VideoFeatDataset(Dataset):
    """Một phần tử = một video: danh sách frame (đặc trưng + target). Chỉ lấy video đã có file .pt trong feats/<kind>/<split>."""

    def __init__(self, index, split, feats_root, kind='gt', limit=0):
        self.dir = os.path.join(feats_root, kind, split)
        self.kind, self.split = kind, split
        have = {fn[:-3] for fn in os.listdir(self.dir) if fn.endswith('.pt')} if os.path.isdir(self.dir) else set()
        self.vids = sorted(v for v in index[split] if v in have)
        if limit:
            self.vids = self.vids[:limit]
        self.frames = {v: index[split][v] for v in self.vids}
        if not self.vids:
            raise FileNotFoundError(f'không có file đặc trưng trong {self.dir} (chạy agyolo.extract_feats trước)')

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        vid = self.vids[i]
        d = torch.load(os.path.join(self.dir, vid + '.pt'), weights_only=False)
        frames = []
        for rec, f in zip(d['frames'], self.frames[vid]):
            assert rec['frame'] == f['frame'], (vid, rec['frame'], f['frame'])
            n = rec['n']
            if self.kind == 'gt':
                att, spa, con, matched = f['att'].astype(np.int64), f['spa'].astype(np.float32), f['con'].astype(np.float32), np.ones(n, bool)
            else:
                att, spa, con, matched = assign_targets(rec, f)
            frames.append(dict(rec=rec, target=make_target(vid, f), att=att, spa=spa, con=con, matched=matched))
        return dict(vid=vid, frames=frames, D=d['D'])


class Clip:
    """Một lô video ghép thành chuỗi token. Mọi tensor token đều có độ dài T."""
    _tensors = ('obj', 'uni', 'per', 'boxes', 'pbox', 'labels', 'scores', 'vid_id', 'fidx', 'frame_id', 'frame_local',
                'slot', 'att', 'spa', 'con', 'matched')

    def to(self, dev):
        for k in self._tensors:
            setattr(self, k, getattr(self, k).to(dev))
        return self


def build_clip(videos):
    """videos: list phần tử của VideoFeatDataset -> (Clip, targets theo frame)."""
    cols = {k: [] for k in Clip._tensors}
    targets, frame_meta = [], []     # frame_meta: (n_tokens, pbox xyxy norm, pscore)
    frame_global = 0
    for vi, v in enumerate(videos):
        for fi, fr in enumerate(v['frames']):
            rec, n = fr['rec'], fr['rec']['n']
            targets.append(fr['target'])
            if n == 0:
                frame_meta.append((0, np.zeros(4, np.float32)))
                frame_global += 1
                continue
            frame_meta.append((n, rec['pbox']))
            cols['obj'].append(rec['obj']); cols['uni'].append(rec['uni']); cols['per'].append(rec['per'][None].expand(n, -1))
            cols['boxes'].append(torch.from_numpy(rec['boxes'])); cols['pbox'].append(torch.from_numpy(rec['pbox'])[None].expand(n, 4))
            cols['labels'].append(torch.from_numpy(rec['labels'].astype(np.int64)))
            cols['scores'].append(torch.from_numpy(rec['scores']))
            for k, val in (('vid_id', vi), ('fidx', fi), ('frame_id', frame_global), ('frame_local', frame_global), ('slot', None)):
                cols[k].append(torch.arange(n) if k == 'slot' else torch.full((n,), val, dtype=torch.long))
            cols['att'].append(torch.from_numpy(fr['att'])); cols['spa'].append(torch.from_numpy(fr['spa']))
            cols['con'].append(torch.from_numpy(fr['con'])); cols['matched'].append(torch.from_numpy(fr['matched']))
            frame_global += 1
    c = Clip()
    D = videos[0]['D']
    empty = dict(obj=torch.zeros(0, D, dtype=torch.float16), uni=torch.zeros(0, D, dtype=torch.float16),
                 per=torch.zeros(0, D, dtype=torch.float16), boxes=torch.zeros(0, 4), pbox=torch.zeros(0, 4),
                 labels=torch.zeros(0, dtype=torch.long), scores=torch.zeros(0), att=torch.zeros(0, dtype=torch.long),
                 spa=torch.zeros(0, N_SPA), con=torch.zeros(0, N_CON), matched=torch.zeros(0, dtype=torch.bool))
    for k in Clip._tensors:
        setattr(c, k, torch.cat(cols[k]) if cols[k] else empty.get(k, torch.zeros(0, dtype=torch.long)))
    c.n_frames = frame_global
    c.frame_meta = frame_meta
    return c, targets


# ----------------------------------------------------------------------------- mô hình
def box_masks(pbox, obox, size=27):
    """pbox, obox [T,4] xyxy chuẩn hoá -> [T,2,size,size]: mặt nạ người/vật trong hệ toạ độ của union box."""
    ub = torch.cat([torch.min(pbox[:, :2], obox[:, :2]), torch.max(pbox[:, 2:], obox[:, 2:])], 1)
    wh = (ub[:, 2:] - ub[:, :2]).clamp(min=1e-4)
    g = (torch.arange(size, device=pbox.device, dtype=pbox.dtype) + 0.5) / size

    def m(b):
        rel = (b - torch.cat([ub[:, :2], ub[:, :2]], 1)) / torch.cat([wh, wh], 1)
        mx = (g[None] >= rel[:, 0:1]) & (g[None] <= rel[:, 2:3])
        my = (g[None] >= rel[:, 1:2]) & (g[None] <= rel[:, 3:4])
        return (my[:, :, None] & mx[:, None, :]).float()
    return torch.stack([m(pbox), m(obox)], 1)


def pair_geom(pbox, obox, eps=1e-4):
    pw, ph = (pbox[:, 2] - pbox[:, 0]).clamp(min=eps), (pbox[:, 3] - pbox[:, 1]).clamp(min=eps)
    ow, oh = (obox[:, 2] - obox[:, 0]).clamp(min=eps), (obox[:, 3] - obox[:, 1]).clamp(min=eps)
    pcx, pcy = (pbox[:, 0] + pbox[:, 2]) / 2, (pbox[:, 1] + pbox[:, 3]) / 2
    ocx, ocy = (obox[:, 0] + obox[:, 2]) / 2, (obox[:, 1] + obox[:, 3]) / 2
    lt = torch.max(pbox[:, :2], obox[:, :2]); rb = torch.min(pbox[:, 2:], obox[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    iou = inter / (pw * ph + ow * oh - inter).clamp(min=eps)
    return torch.stack([(ocx - pcx) / pw, (ocy - pcy) / ph, torch.log(ow / pw), torch.log(oh / ph), iou, ow * oh], 1)


def sincos(pos, d):
    i = torch.arange(d // 2, device=pos.device, dtype=torch.float32)
    ang = pos[:, None].float() / torch.pow(10000.0, 2 * i / d)[None]
    return torch.cat([ang.sin(), ang.cos()], 1)


class STTran(nn.Module):
    def __init__(self, feat_dim, d=512, heads=8, ff=1024, n_spatial=1, n_temporal=3, window=2, drop=0.1, n_cls=N_OBJ):
        super().__init__()
        self.d, self.window = d, window
        proj = lambda: nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256), nn.ReLU())
        self.f_per, self.f_obj, self.f_uni = proj(), proj(), proj()
        self.mask_net = nn.Sequential(nn.Conv2d(2, 32, 5, 2, 2), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(3, 2, 1),
                                      nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.Flatten(),
                                      nn.Linear(64 * 7 * 7, 256), nn.ReLU())
        self.cls_emb = nn.Embedding(n_cls, 64)
        self.geo = nn.Sequential(nn.Linear(6, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 * 4 + 64 + 64, d), nn.ReLU(), nn.LayerNorm(d))
        mk = lambda n: nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, activation='gelu',
                                                                         batch_first=True, norm_first=True),
                                             n, enable_nested_tensor=False)
        self.spatial, self.temporal = mk(n_spatial), mk(n_temporal)
        self.att_head, self.spa_head, self.con_head = nn.Linear(d, N_ATT), nn.Linear(d, N_SPA), nn.Linear(d, N_CON)

    def forward(self, c):
        T = c.obj.shape[0]
        if T == 0:
            z = c.obj.new_zeros(0, 1).float()
            return dict(att=z.new_zeros(0, N_ATT), spa=z.new_zeros(0, N_SPA), con=z.new_zeros(0, N_CON))
        masks = box_masks(c.pbox.float(), c.boxes.float())
        x = torch.cat([self.f_per(c.per.float()), self.f_obj(c.obj.float()), self.f_uni(c.uni.float()),
                       self.mask_net(masks), self.cls_emb(c.labels.clamp(0, N_OBJ - 1)),
                       self.geo(pair_geom(c.pbox.float(), c.boxes.float()))], 1)
        x = self.fuse(x)[None]                                           # [1,T,d]
        same_frame = c.frame_id[:, None] == c.frame_id[None]
        x = self.spatial(x, mask=~same_frame)
        diff = c.fidx[:, None] - c.fidx[None]
        allowed = (c.vid_id[:, None] == c.vid_id[None]) & (diff >= 0) & (diff < self.window)
        x = x + sincos(c.fidx, self.d)[None].to(x.dtype)
        x = self.temporal(x, mask=~allowed)[0]
        return dict(att=self.att_head(x), spa=self.spa_head(x), con=self.con_head(x))


# ----------------------------------------------------------------------------- loss
def multilabel_loss(logits, tgt, pw=None, focal=True, gamma=2.0, alpha=0.25):
    if not focal:
        return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction='sum')
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction='none')
    pt = p * tgt + (1 - p) * (1 - tgt)
    at = alpha * tgt + (1 - alpha) * (1 - tgt)
    return (at * (1 - pt) ** gamma * ce).sum()


def relation_loss(out, c, pw_spa=None, pw_con=None, focal=True, w=(1.0, 1.0, 1.0)):
    m = c.matched
    P = int(m.sum())
    z = out['att'].sum() * 0
    if P == 0:
        return z, dict(att=0.0, spa=0.0, con=0.0)
    la = F.cross_entropy(out['att'][m], c.att[m], ignore_index=-1, reduction='sum') / P
    ls = multilabel_loss(out['spa'][m], c.spa[m], pw_spa, focal) / P
    lc = multilabel_loss(out['con'][m], c.con[m], pw_con, focal) / P
    return w[0] * la + w[1] * ls + w[2] * lc, dict(att=float(la), spa=float(ls), con=float(lc))
