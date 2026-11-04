"""Adapter Action Genome: đọc ag_index.pkl (tools/build_index.py) + thư mục frame -> (image, target) cho criterion.py.

Target: person_box cxcywh chuẩn hóa | obj_labels | obj_boxes | att | spa | con | size=(h,w) ẢNH GỐC | has_person.
Box chuẩn hóa theo kích thước ảnh gốc nên không đổi khi resize; collate_fn pad xuống dưới/phải.
"""
import os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .sampling import rfs_repeat_factors
from .constants import N_OBJ, N_ATT, N_SPA, N_CON

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _cxcywh(xyxy, w, h):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]


class AGDataset(Dataset):
    def __init__(self, index, split, frames_root, train=False, short=384, max_side=640,
                 frames_per_video=0, rfs=False, seed=0, flip=True, jitter=True, limit=0):
        self.root, self.train, self.short, self.max_side = frames_root, train, short, max_side
        self.fpv, self.rfs, self.flip = frames_per_video, rfs and train, flip and train
        self.jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.05) if (jitter and train) else None
        self.rng = random.Random(seed)
        self.videos = {v: frs for v, frs in index[split].items()}
        self.all = [(v, f) for v, frs in self.videos.items() for f in frs]
        if limit > 0 and limit < len(self.all):     # tập con cố định (vd dev300 để theo dõi lúc train)
            self.all = random.Random(12345).sample(self.all, limit)
        if self.rfs:   # presence 61 lớp: object(35) + attention(3) + spatial(6) + contacting(17)
            pres = np.zeros((len(self.all), N_OBJ + N_ATT + N_SPA + N_CON), bool)
            for i, (_, f) in enumerate(self.all):
                pres[i, np.unique(f['labels'])] = True
                pres[i, N_OBJ + np.unique(f['att'])] = True
                pres[i, N_OBJ + N_ATT + np.where(f['spa'].any(0))[0]] = True
                pres[i, N_OBJ + N_ATT + N_SPA + np.where(f['con'].any(0))[0]] = True
            self.ri, _, _ = rfs_repeat_factors(pres, t=0.1, cap=10.0)
        self.items = []
        self.resample()

    def resample(self):
        """Gọi đầu mỗi epoch: chọn lại frame/video và (nếu bật) lặp frame theo RFS."""
        if self.fpv > 0:
            pool = []
            for v, frs in self.videos.items():
                pool += [(v, f) for f in self.rng.sample(frs, min(self.fpv, len(frs)))]
        else:
            pool = list(self.all)
        if self.rfs:
            r = {id(f): self.ri[i] for i, (_, f) in enumerate(self.all)}
            out = []
            for v, f in pool:
                x = r[id(f)]
                out += [(v, f)] * (int(x) + (self.rng.random() < x - int(x)))
            pool = out
        self.items = pool

    def expected_len(self):
        n = len(self.all) if self.fpv <= 0 else sum(min(self.fpv, len(f)) for f in self.videos.values())
        return int(n * (float(self.ri.mean()) if self.rfs else 1.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, f = self.items[i]
        im = Image.open(os.path.join(self.root, vid, f['frame'])).convert('RGB')
        w, h = f['size']
        if im.size != (w, h):                       # ảnh dump khác kích thước nhãn -> đổi tỉ lệ box
            sx, sy = im.size[0] / w, im.size[1] / h
            f = dict(f, person=f['person'] * np.array([sx, sy, sx, sy], np.float32),
                     boxes=f['boxes'] * np.array([sx, sy, sx, sy], np.float32))
            w, h = im.size
        pb = np.array(f['person']) if f['has_person'] else np.array([0.25, 0.25, 0.75, 0.75], np.float32)
        ob = np.array(f['boxes']).reshape(-1, 4)
        person = torch.tensor(_cxcywh(pb, w, h))
        boxes = torch.tensor([_cxcywh(b, w, h) for b in ob], dtype=torch.float32).reshape(-1, 4)
        if self.flip and self.rng.random() < 0.5:   # lật ngang: chỉ đổi cx
            im = TF.hflip(im)
            person[0] = 1 - person[0]; boxes[:, 0] = 1 - boxes[:, 0]
        if self.jitter is not None:
            im = self.jitter(im)
        s = min(self.short / min(w, h), self.max_side / max(w, h))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        x = TF.normalize(TF.to_tensor(im), MEAN, STD)
        t = dict(person_box=person, size=(h, w), has_person=bool(f['has_person']), vid=vid, frame=f['frame'],
                 obj_labels=torch.as_tensor(f['labels'], dtype=torch.long), obj_boxes=boxes,
                 att=torch.as_tensor(f['att'], dtype=torch.long),
                 spa=torch.as_tensor(f['spa'], dtype=torch.float32),
                 con=torch.as_tensor(f['con'], dtype=torch.float32))
        return x, t


def label_counts(index, split):
    """Đếm dương theo lớp trên train -> (spa_counts [6], con_counts [17], n_instance) cho pos_weight_from_counts."""
    spa = np.zeros(N_SPA); con = np.zeros(N_CON); n = 0
    for frs in index[split].values():
        for f in frs:
            spa += f['spa'].sum(0); con += f['con'].sum(0); n += len(f['labels'])
    return spa, con, n
