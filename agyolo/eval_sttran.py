"""Đánh giá STTran trên đặc trưng YOLO, DÙNG LẠI evaluator của agsgg (cùng R@K / mR@K / phân rã lỗi / mAP50)
nên số liệu so sánh trực tiếp được với baseline agsgg.

  python -m agyolo.eval_sttran --ckpt runs/sttran/best.pth --index ag_index.pkl --feats feats --kind gt  --split test --out results/sttran_predcls
  python -m agyolo.eval_sttran --ckpt runs/sttran/best.pth --index ag_index.pkl --feats feats --kind det --split test --out results/sttran_sgdet

--kind gt  = PredCls (box + nhãn vật thật, chỉ đo phần quan hệ);  --kind det = SGDet (box + nhãn do YOLO dự đoán).
Kết quả (result.json, per_frame.csv, preds.pkl) cùng định dạng agsgg.evaluate nên make_report_assets / visualize_preds dùng được.
"""
import argparse, csv, json, os, pickle

import torch
from torch.utils.data import DataLoader

from agsgg.constants import N_ATT, N_SPA, N_CON, N_OBJ
from agsgg.evaluate import evaluate, KS, STAGES
from agsgg.runinfo import run_info
from .sttran import STTran, VideoFeatDataset, build_clip


class Pack:
    """Bọc Clip để giả làm tensor (evaluate() gọi x.to(device))."""

    def __init__(self, clip):
        self.clip = clip

    def to(self, dev):
        self.clip = self.clip.to(dev)
        return self


def collate_eval(videos):
    clip, targets = build_clip(videos)
    return Pack(clip), torch.zeros(1), targets


def xyxy_to_cxcywh(b):
    return torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                        b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)


class Adapter:
    """Cho STTran trông như model của agsgg: trả dict pred_logits / pred_boxes / person_box / att,spa,con_logits theo frame."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, x, mask):
        c = x.clip
        dev = c.obj.device
        nf = c.n_frames
        N = max([n for n, _ in c.frame_meta] + [1])
        out = self.model(c)
        logits = torch.full((nf, N, N_OBJ + 1), -1e4, device=dev)
        logits[..., N_OBJ] = 0.0
        boxes = torch.zeros(nf, N, 4, device=dev)
        person = torch.zeros(nf, 4, device=dev)
        att = torch.zeros(nf, N, N_ATT, device=dev); spa = torch.zeros(nf, N, N_SPA, device=dev)
        con = torch.zeros(nf, N, N_CON, device=dev)
        T = c.obj.shape[0]
        if T:
            fl, sl = c.frame_local, c.slot
            s = c.scores.float().clamp(1e-4, 1 - 1e-6)
            row = torch.full((T, N_OBJ + 1), -1e4, device=dev)
            row[torch.arange(T, device=dev), c.labels.clamp(0, N_OBJ - 1)] = s.log()
            row[:, N_OBJ] = (1 - s).log()
            logits[fl, sl] = row
            boxes[fl, sl] = xyxy_to_cxcywh(c.boxes.float())
            att[fl, sl], spa[fl, sl], con[fl, sl] = out['att'], out['spa'], out['con']
        for f, (n, pb) in enumerate(c.frame_meta):
            if n:
                person[f] = xyxy_to_cxcywh(torch.as_tensor(pb, device=dev).float())
        return dict(pred_logits=logits, pred_boxes=boxes, person_box=person,
                    att_logits=att, spa_logits=spa, con_logits=con)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = STTran(**ck['cfg']).to(device)
    model.load_state_dict(ck['model'])
    return model, ck


def make_loader(index, split, feats, kind, bs_videos=4, workers=0, limit=0):
    ds = VideoFeatDataset(index, split, feats, kind=kind, limit=limit)
    return DataLoader(ds, batch_size=bs_videos, shuffle=False, collate_fn=collate_eval, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--feats', default='feats')
    ap.add_argument('--kind', default='gt', choices=['gt', 'det'])
    ap.add_argument('--split', default='test', choices=['dev', 'test', 'train'])
    ap.add_argument('--out', default='results/sttran')
    ap.add_argument('--bs', type=int, default=4, help='số video mỗi lô')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='>0: chỉ k video đầu')
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = pickle.load(open(a.index, 'rb'))
    model, ck = load_model(a.ckpt, a.device)
    dl = make_loader(index, a.split, a.feats, a.kind, a.bs, a.workers, a.limit)
    res, per_frame, saved = evaluate(Adapter(model), dl, a.device, save_preds=a.save_preds)
    res['mode'] = 'PredCls' if a.kind == 'gt' else 'SGDet'
    res['args'] = vars(a); res['n_params'] = sum(p.numel() for p in model.parameters())
    res['run_info'] = run_info({'checkpoint': a.ckpt, 'index': a.index})
    print(f"[{res['mode']}] {a.split}: {res['frames_evaluated']} frame (bỏ {res['frames_skipped_no_person']} không có người), "
          f"{res['seconds'] / 60:.1f} phút")
    for m, name in (('with', 'With Constraint'), ('no', 'No Constraint')):
        for iou in ('iou0.5', 'iou0.3'):
            r = res[m][iou]
            print(f'[{name} {iou}] ' + '  '.join(f"R@{k}={r[f'R@{k}']:.1f}" for k in KS) + '   ' +
                  '  '.join(f"mR@{k}={r[f'mR@{k}']:.1f}" for k in KS))
    fb = res['failure_breakdown_with_iou0.5_K20']['all']
    print('Phân rã lỗi (With, IoU0.5, K=20): ' + ', '.join(f'{s}={fb[s]:.1f}%' for s in STAGES),
          f"| trùng lặp top-20: {res['duplicate_rate_top20_with_pct']:.1f}% | mAP50 thực thể: {res['entity']['mAP50']:.1f}")
    json.dump(res, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, 'per_frame.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['video', 'frame', 'n_gt', 'R@10', 'R@20', 'R@50']); w.writerows(per_frame)
    if saved:
        pickle.dump(saved, open(os.path.join(a.out, 'preds.pkl'), 'wb'))
    print('đã lưu vào', a.out)


if __name__ == '__main__':
    main()
