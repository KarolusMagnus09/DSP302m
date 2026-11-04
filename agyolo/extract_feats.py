"""Bước 3: chạy YOLO đã fine-tune trên frame, lưu box/nhãn + đặc trưng RoI cho STTran (mỗi video một file .pt).

  python -m agyolo.extract_feats --weights runs/yolo/ag/weights/best.pt --index ag_index.pkl --frames frames \
      --split train --kind gt  --video_frac 0.3 --out feats
  python -m agyolo.extract_feats ... --split test --kind both --max_videos 150 --out feats

--kind gt  : dùng box NHÃN THẬT (huấn luyện + đánh giá PredCls). Đặc trưng vẫn do YOLO tạo ra.
--kind det : dùng box YOLO dự đoán (đánh giá SGDet). Người = detection 'person' điểm cao nhất; vật = top --max_obj.
--kind both: cả hai trong cùng một lượt chạy.

Đặc trưng: hook vào đầu vào của head Detect (3 feature map P3/P4/P5 của neck), RoIAlign 3x3 rồi trung bình ->
vector mỗi tầng, nối lại (D = tổng kênh, ví dụ yolo26s = 896). Mỗi cặp (người, vật) có 3 vector:
người, vật, union box. Lưu float16. Ảnh được letterbox vuông imgsz x imgsz (đệm xám 114, căn giữa).
Chạy lại được: video đã có file sẽ bị bỏ qua.
"""
import argparse, os, pickle, random, time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import roi_align

from agsgg.progress import tqdm, write


class FrameDS(Dataset):
    def __init__(self, items, root, S):
        self.items, self.root, self.S = items, root, S

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, f = self.items[i][1], self.items[i][2]
        S = self.S
        im = None
        try:
            im = cv2.imdecode(np.fromfile(os.path.join(self.root, vid, f['frame']), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass
        ok = im is not None
        if not ok:
            im = np.full((f['size'][1], f['size'][0], 3), 114, np.uint8)
        h, w = im.shape[:2]
        r = S / max(h, w)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((S, S, 3), 114, np.uint8)
        l, t = (S - nw) // 2, (S - nh) // 2
        canvas[t:t + nh, l:l + nw] = im
        x = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1])).permute(2, 0, 1)
        return x, (i, w, h, r, l, t, ok)


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def roi_feats(fmaps, S, bidx, boxes):
    """fmaps: list [B,C,h,w]; bidx [R]; boxes [R,4] xyxy toạ độ canvas letterbox -> [R, sum(C)]."""
    rois = torch.cat([bidx[:, None].float(), boxes.float()], 1)
    out = []
    for fm in fmaps:
        stride = S / fm.shape[-1]
        o = roi_align(fm.float(), rois, output_size=3, spatial_scale=1.0 / stride, sampling_ratio=2, aligned=True)
        out.append(o.mean((2, 3)))
    return torch.cat(out, 1)


def union(pb, ob):
    return torch.cat([torch.min(pb[:, :2], ob[:, :2]), torch.max(pb[:, 2:], ob[:, 2:])], 1)


def select_dets(res_b, max_obj):
    """res_b: tensor [n,6] (x1,y1,x2,y2,conf,cls) -> (person [1,5] hoặc None, vật [<=max_obj,6])."""
    if res_b.numel() == 0:
        return None, res_b.new_zeros(0, 6)
    cls = res_b[:, 5].long()
    pm = cls == 0
    person = None
    if pm.any():
        pr = res_b[pm]
        person = pr[pr[:, 4].argmax()]
    ob = res_b[~pm]
    ob = ob[ob[:, 4].argsort(descending=True)][:max_obj]
    return person, ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--kind', default='gt', choices=['gt', 'det', 'both'])
    ap.add_argument('--out', default='feats')
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--conf', type=float, default=0.1)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--max_obj', type=int, default=12)
    ap.add_argument('--video_frac', type=float, default=1.0, help='lấy ngẫu nhiên tỉ lệ video (giữ nguyên chuỗi frame của video)')
    ap.add_argument('--max_videos', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--half', action='store_true', help='suy luận fp16 (GPU có tensor core); mặc định fp32 cho an toàn')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    from ultralytics import YOLO
    S, dev = a.imgsz, a.device
    kinds = ['gt', 'det'] if a.kind == 'both' else [a.kind]
    index = pickle.load(open(a.index, 'rb'))
    vids = sorted(index[a.split])
    if a.video_frac < 1:
        vids = sorted(random.Random(a.seed).sample(vids, max(1, round(a.video_frac * len(vids)))))
    if a.max_videos:
        vids = vids[:a.max_videos]
    outs = {k: os.path.join(a.out, k, a.split) for k in kinds}
    for p in outs.values():
        os.makedirs(p, exist_ok=True)
    todo = [v for v in vids if not all(os.path.exists(os.path.join(outs[k], v + '.pt')) for k in kinds)]
    write(f'{a.split}: {len(vids)} video được chọn, {len(todo)} cần trích ({len(vids) - len(todo)} đã có)')
    items = [(vi, v, f) for vi, v in enumerate(todo) for f in index[a.split][v]]
    if not items:
        return
    total = {v: len(index[a.split][v]) for v in todo}

    yolo = YOLO(a.weights)
    store = {}
    head = yolo.model.model[-1]
    head.register_forward_pre_hook(lambda m, inp: store.__setitem__('f', list(inp[0])))
    half = a.half and dev.startswith('cuda')
    dl = DataLoader(FrameDS(items, a.frames, S), batch_size=a.bs, shuffle=False, collate_fn=collate,
                    num_workers=a.workers)
    buf = {k: {} for k in kinds}
    done_cnt = {}
    t0, n_img, D = time.time(), 0, None

    for x, metas in tqdm(dl, desc='trích đặc trưng', unit='batch'):
        xb = x.to(dev).float() / 255
        res = yolo.predict(xb, imgsz=S, conf=a.conf if 'det' in kinds else 0.25, iou=a.iou, max_det=100,
                           half=half, device=dev, verbose=False)
        fmaps = store['f']
        rows_b, rows_box, plan = [], [], []   # plan: (kind, item_i, meta, sizes, start)
        recs = []
        for b, (i, w, h, r, l, t, ok) in enumerate(metas):
            _, vid, f = items[i]
            sw, sh = f['size']
            for kind in kinds:
                if kind == 'gt':
                    sx, sy = w / sw, h / sh
                    sc = np.array([sx, sy, sx, sy], np.float32)
                    pbox = (f['person'] * sc) if f['has_person'] else None
                    obox = f['boxes'] * sc
                    labels, scores = f['labels'].copy(), np.ones(len(f['labels']), np.float32)
                    pscore = 1.0
                else:
                    person, ob = select_dets(res[b].boxes.data.cpu(), a.max_obj)
                    pbox = None if person is None else person[:4].numpy() / np.array([r, r, r, r]) - np.array([l, t, l, t]) / r
                    pscore = 0.0 if person is None else float(person[4])
                    obox = ob[:, :4].numpy() / r - np.array([l, t, l, t]) / r
                    labels = (ob[:, 5].long() - 1).numpy()
                    scores = ob[:, 4].numpy().astype(np.float32)
                n = len(obox)
                rec = dict(frame=f['frame'], kind=kind, vid=vid, n=0, ok=ok)
                if pbox is not None and n > 0:
                    # toạ độ ảnh gốc -> canvas letterbox
                    cs = np.array([r, r, r, r], np.float32); off = np.array([l, t, l, t], np.float32)
                    pb_c = torch.tensor(pbox * cs + off, dtype=torch.float32)[None]
                    ob_c = torch.tensor(obox * cs + off, dtype=torch.float32)
                    pb_rep = pb_c.expand(n, 4)
                    boxes = torch.cat([pb_c, ob_c, union(pb_rep, ob_c)], 0)
                    rows_b.append(torch.full((n * 2 + 1,), b)); rows_box.append(boxes)
                    plan.append((len(recs), n))
                    nz = np.array([w, h, w, h], np.float32)
                    rec.update(n=n, boxes=np.clip(obox / nz, 0, 1).astype(np.float32),
                               labels=labels.astype(np.int16), scores=scores,
                               pbox=np.clip(pbox / nz, 0, 1).astype(np.float32), pscore=float(pscore))
                else:
                    plan.append((len(recs), 0))
                recs.append((kind, vid, rec))
        if rows_b:
            F_ = roi_feats(fmaps, S, torch.cat(rows_b).to(dev), torch.cat(rows_box).to(dev)).half().cpu()
            D = F_.shape[1]
            p = 0
            for (ri, n) in plan:
                if n == 0:
                    continue
                blk = F_[p:p + 2 * n + 1]; p += 2 * n + 1
                recs[ri][2].update(per=blk[0].clone(), obj=blk[1:1 + n].clone(), uni=blk[1 + n:].clone())
        for kind, vid, rec in recs:
            buf[kind].setdefault(vid, []).append(rec)
        for kind in kinds:
            for vid in list(buf[kind]):
                if len(buf[kind][vid]) == total[vid]:
                    torch.save(dict(vid=vid, kind=kind, D=D, frames=buf[kind].pop(vid)),
                               os.path.join(outs[kind], vid + '.pt'))
        n_img += len(metas)
    write(f'xong: {n_img} frame trong {(time.time() - t0) / 60:.1f} phút ({n_img / (time.time() - t0):.1f} ảnh/s) -> {a.out}')


if __name__ == '__main__':
    main()
