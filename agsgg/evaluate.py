"""Đánh giá SGDet kiểu STTran và lưu MỌI số liệu cần cho báo cáo.

  python -m agsgg.evaluate --ckpt runs/agsgg/epoch5.pth --index ag_index.pkl --frames <frames> --split test --out results/test_full [--save_preds]

Với AG-filtersmall: tạo thêm ag_index_small.pkl bằng build_index --filter_small rồi chạy lại với --index ag_index_small.pkl.

Một bộ ba GT (người, vật, vị từ) được tìm thấy nếu trong top-K có bộ ba cùng nhãn vật, cùng vị từ, IoU(box người) và IoU(box vật) >= ngưỡng.
R@K = trung bình recall theo frame. mR@K = trung bình theo vị từ (mỗi vị từ: trung bình recall trên các frame có vị từ đó).
Frame không có bbox người bị bỏ (giống STTran). Kết quả lưu trong --out:
  result.json   R@K/mR@K (With/No x IoU 0.5/0.3), recall theo nhóm và theo lớp (kèm n GT), phân rã lỗi, tỉ lệ trùng lặp top-20, AP50 thực thể, thông tin tái lập
  per_frame.csv recall từng frame (With, IoU 0.5) để chọn ví dụ tốt/xấu
  preds.pkl     top-50 dự đoán (With) mỗi frame, nếu có --save_preds (dùng cho tools/visualize_preds.py)
"""
import argparse, csv, json, os, pickle, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou

from .constants import PREDICATES, OBJ_CLASSES
from .data import collate_fn
from .model import AGSGG
from .postprocess import postprocess
from .progress import tqdm

KS = (10, 20, 50)
GROUPS = {'attention': (0, 3), 'spatial': (3, 9), 'contacting': (9, 26)}
STAGES = ['loc', 'label', 'predicate', 'rank', 'found']   # khâu thất bại đầu tiên (found = tìm thấy)


def gt_triplets(t):
    """-> (obj_idx [G], pred_idx [G]) mọi (vật, vị từ) dương: attention 1 nhãn, spatial/contacting đa nhãn."""
    oi, pi = [], []
    for j in range(len(t['obj_labels'])):
        if int(t['att'][j]) >= 0:
            oi.append(j); pi.append(int(t['att'][j]))
        for k in torch.nonzero(t['spa'][j] > 0.5).flatten().tolist():
            oi.append(j); pi.append(3 + k)
        for k in torch.nonzero(t['con'][j] > 0.5).flatten().tolist():
            oi.append(j); pi.append(9 + k)
    return torch.tensor(oi, dtype=torch.long), torch.tensor(pi, dtype=torch.long)


def match_info(pred, t, thr):
    """Ma trận bool [G, n] trên toàn bộ n dự đoán (đã sắp giảm dần theo điểm): loc (cả 2 box), lab, prd."""
    oi, pi = gt_triplets(t)
    h, w = t['size']
    sc = torch.tensor([w, h, w, h], dtype=torch.float32)
    gb = box_convert(t['obj_boxes'][oi], 'cxcywh', 'xyxy') * sc
    gp = box_convert(t['person_box'][None], 'cxcywh', 'xyxy') * sc
    ob, sb = pred['obj_boxes'].float().cpu(), pred['sub_box'].float().cpu()
    loc = (box_iou(gb, ob) >= thr) & (box_iou(gp, sb)[0] >= thr)[None]
    lab = (t['obj_labels'][oi][:, None] + 1) == pred['obj_labels'].cpu()[None]
    prd = pi[:, None] == pred['predicates'].cpu()[None]
    return oi, pi, loc, lab, prd


def dup_rate_top20(pred):
    """Tỉ lệ vị trí trong top-20 là bản trùng của một dự đoán xếp trên (cùng nhãn vật, vị từ, IoU box vật >= 0.5)."""
    n = min(20, len(pred['scores']))
    if n < 2:
        return 0, n
    ob, lb, pr = pred['obj_boxes'][:n].float().cpu(), pred['obj_labels'][:n].cpu(), pred['predicates'][:n].cpu()
    same = (lb[:, None] == lb[None]) & (pr[:, None] == pr[None]) & (box_iou(ob, ob) >= 0.5)
    return int(torch.tril(same, -1).any(1).sum()), n


class EntityAP:
    """AP50 theo lớp của entity decoder (không NMS, kiểu DETR) + tỉ lệ IoU>=0.5 của box người."""

    def __init__(self, n=len(OBJ_CLASSES)):
        self.sc = [[] for _ in range(n)]; self.tp = [[] for _ in range(n)]; self.ngt = [0] * n; self.person = []

    def add(self, logits, boxes, person_box, t):
        h, w = t['size']
        sc = torch.tensor([w, h, w, h], dtype=torch.float32)
        prob = logits.float().cpu().softmax(-1)[:, :-1]
        score, lab = prob.max(-1)
        pb = box_convert(boxes.float().cpu(), 'cxcywh', 'xyxy') * sc
        gl = t['obj_labels']; gb = box_convert(t['obj_boxes'], 'cxcywh', 'xyxy') * sc
        for c in set(lab.tolist()) | set(gl.tolist()):
            gm = gl == c
            self.ngt[c] += int(gm.sum())
            pm = lab == c
            if not pm.any():
                continue
            ps, pbx = score[pm], pb[pm]
            order = ps.argsort(descending=True)
            gbx = gb[gm]
            ious = box_iou(pbx[order], gbx) if len(gbx) else None
            used = torch.zeros(len(gbx), dtype=torch.bool)
            for k, idx in enumerate(order.tolist()):
                ok = False
                if ious is not None:
                    iou = ious[k].clone(); iou[used] = -1
                    j = int(iou.argmax())
                    if iou[j] >= 0.5:
                        used[j] = True; ok = True
                self.sc[c].append(float(ps[idx])); self.tp[c].append(ok)
        if t.get('has_person', True):
            pp = box_convert(person_box[None].float().cpu(), 'cxcywh', 'xyxy') * sc
            gp = box_convert(t['person_box'][None], 'cxcywh', 'xyxy') * sc
            self.person.append(float(box_iou(pp, gp)[0, 0] >= 0.5))

    def result(self):
        ap = {}
        for c, name in enumerate(OBJ_CLASSES):
            if self.ngt[c] == 0:
                continue
            if not self.sc[c]:
                ap[name] = 0.0; continue
            o = np.argsort(-np.array(self.sc[c])); tp = np.array(self.tp[c])[o].astype(float)
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec, prec = ctp / self.ngt[c], ctp / np.maximum(ctp + cfp, 1e-9)
            mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap[name] = 100 * float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
        return dict(AP50_per_class=ap, mAP50=float(np.mean(list(ap.values()))) if ap else 0.0,
                    person_iou50_rate=100 * float(np.mean(self.person)) if self.person else 0.0,
                    n_gt_per_class={OBJ_CLASSES[c]: n for c, n in enumerate(self.ngt) if n})


@torch.no_grad()
def evaluate(model, dl, device, modes=('with', 'no'), ious=(0.5, 0.3), extra=True, save_preds=False):
    model.eval()
    P = len(PREDICATES)
    rec = {(m, i): {k: [] for k in KS} for m in modes for i in ious}
    per = {(m, i): {k: [[] for _ in range(P)] for k in KS} for m in modes for i in ious}
    grp = {(m, i): {k: {g: [] for g in GROUPS} for k in KS} for m in modes for i in ious}
    n_gt_pred = np.zeros(P, dtype=np.int64); n_gt_frames = np.zeros(P, dtype=np.int64)
    stage = {g: {s: 0 for s in STAGES} for g in list(GROUPS) + ['all']}
    dup = [0, 0]; ap = EntityAP() if extra else None
    per_frame, saved = [], []
    n_frames = n_noperson = 0
    t0 = time.time()
    for x, mask, tgts in tqdm(dl, desc='đánh giá', unit='batch', leave=False):
        out = model(x.to(device), mask.to(device))
        sizes = [t['size'] for t in tgts]
        preds = {m: postprocess(out, sizes, constraint=m, topk=100) for m in modes}
        for b, t in enumerate(tgts):
            if not t.get('has_person', True):
                n_noperson += 1
                continue
            n_frames += 1
            if extra:
                ap.add(out['pred_logits'][b], out['pred_boxes'][b], out['person_box'][b], t)
            for m in modes:
                pr = preds[m][b]
                for iou in ious:
                    oi, pi, loc, lab, prd = match_info(pr, t, iou)
                    full = loc & lab & prd
                    for k in KS:
                        hit = full[:, :k].any(1)
                        rec[(m, iou)][k].append(hit.float().mean().item())
                        for p in pi.unique().tolist():
                            per[(m, iou)][k][p].append(hit[pi == p].float().mean().item())
                        for g, (lo, hi) in GROUPS.items():
                            sel = (pi >= lo) & (pi < hi)
                            if sel.any():
                                grp[(m, iou)][k][g].append(hit[sel].float().mean().item())
                    if m == modes[0] and iou == ious[0]:
                        if extra:
                            for p in pi.tolist():
                                n_gt_pred[p] += 1
                            for p in pi.unique().tolist():
                                n_gt_frames[p] += 1
                        h20 = full[:, :20].any(1)
                        per_frame.append((t.get('vid', ''), t.get('frame', ''), len(oi),
                                          *[float(full[:, :k].any(1).float().mean()) for k in KS]))
                        if extra and m == 'with' and iou == 0.5:
                            e_loc = loc.any(1); e_lab = (loc & lab).any(1); e_prd = full.any(1)
                            st = torch.full((len(pi),), 4, dtype=torch.long)
                            st[~h20] = 3; st[~e_prd] = 2; st[~e_lab] = 1; st[~e_loc] = 0
                            for s, p in zip(st.tolist(), pi.tolist()):
                                stage['all'][STAGES[s]] += 1
                                for g, (lo, hi) in GROUPS.items():
                                    if lo <= p < hi:
                                        stage[g][STAGES[s]] += 1
                if extra and m == 'with':
                    d, n = dup_rate_top20(pr); dup[0] += d; dup[1] += n
                if save_preds and m == 'with':
                    n = min(50, len(pr['scores']))
                    saved.append(dict(vid=t.get('vid', ''), frame=t.get('frame', ''), size=t['size'],
                                      **{k: pr[k][:n].cpu().numpy() for k in
                                         ('scores', 'predicates', 'obj_labels', 'obj_boxes', 'sub_box')}))
    res = dict(frames_evaluated=n_frames, frames_skipped_no_person=n_noperson, seconds=time.time() - t0)
    for m in modes:
        for iou in ious:
            r = {}
            for k in KS:
                pk = [float(np.mean(v)) for v in per[(m, iou)][k] if v]
                r[f'R@{k}'] = 100 * float(np.mean(rec[(m, iou)][k])) if rec[(m, iou)][k] else 0.0
                r[f'mR@{k}'] = 100 * float(np.mean(pk)) if pk else 0.0
                if extra:
                    r[f'per_predicate_R@{k}'] = {PREDICATES[p]: dict(recall=100 * float(np.mean(v)), n_frames=int(n_gt_frames[p]),
                                                 n_gt=int(n_gt_pred[p])) for p, v in enumerate(per[(m, iou)][k]) if v}
                    r[f'group_R@{k}'] = {g: 100 * float(np.mean(v)) for g, v in grp[(m, iou)][k].items() if v}
            res.setdefault(m, {})[f'iou{iou}'] = r
    if extra:
        tot = {g: max(sum(c.values()), 1) for g, c in stage.items()}
        res['failure_breakdown_with_iou0.5_K20'] = {g: {s: 100 * c[s] / tot[g] for s in STAGES} | {'n_gt': tot[g]}
                                                    for g, c in stage.items()}
        res['duplicate_rate_top20_with_pct'] = 100 * dup[0] / max(dup[1], 1)
        res['entity'] = ap.result()
        res['n_gt_triplets_total'] = int(n_gt_pred.sum())
    return res, per_frame, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--frames', required=True); ap.add_argument('--split', default='test', choices=['dev', 'test', 'train'])
    ap.add_argument('--out', default='results/test')
    ap.add_argument('--bs', type=int, default=8); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--short', type=int, default=384); ap.add_argument('--max_side', type=int, default=640)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='>0: chỉ đánh giá k video đầu (chạy thử cho nhanh)')
    ap.add_argument('--save_preds', action='store_true', help='lưu top-50 dự đoán (With) để vẽ ví dụ định tính')
    a = ap.parse_args()
    from .ag_dataset import AGDataset
    from .runinfo import run_info
    os.makedirs(a.out, exist_ok=True)
    index = pickle.load(open(a.index, 'rb'))
    if a.limit:
        index[a.split] = dict(list(index[a.split].items())[:a.limit])
    ds = AGDataset(index, a.split, a.frames, train=False, short=a.short, max_side=a.max_side)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, collate_fn=collate_fn, num_workers=a.workers)
    model = AGSGG(backbone_pretrained=False).to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device)['model'])
    res, per_frame, saved = evaluate(model, dl, a.device, save_preds=a.save_preds)
    res['args'] = vars(a); res['n_params'] = sum(p.numel() for p in model.parameters())
    res['run_info'] = run_info({'checkpoint': a.ckpt, 'index': a.index})
    print(f"{a.split}: {res['frames_evaluated']} frame (bỏ {res['frames_skipped_no_person']} frame không có người), "
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
