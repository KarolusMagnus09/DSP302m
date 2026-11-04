"""Huấn luyện STTran trên đặc trưng YOLO đã trích (nhanh: không còn backbone/ảnh trong vòng lặp).

  python -m agyolo.train_sttran --index ag_index.pkl --feats feats --epochs 12 --bs 4 --out runs/sttran

--train_kind gt : học với box/nhãn vật thật (như PredCls). Khuyến nghị cho lần đầu.
--train_kind det: học với box YOLO phát hiện; chỉ tính loss trên vật khớp GT (cùng lớp, IoU>=0.5).
Mỗi epoch đo R@K trên dev nếu có feats/<dev_kind>/dev (lưu best.pth theo R@20 With, IoU 0.5).
Log: train_log.csv, dev_log.csv, config.json (tương thích agsgg.tools.make_report_assets).
"""
import argparse, copy, csv, json, math, os, pickle, time

import torch
from torch.utils.data import DataLoader

from agsgg.ag_dataset import label_counts
from agsgg.criterion import pos_weight_from_counts
from agsgg.evaluate import evaluate, KS
from agsgg.progress import tqdm, write
from agsgg.runinfo import run_info
from .eval_sttran import Adapter, make_loader
from .sttran import STTran, VideoFeatDataset, build_clip, relation_loss


def collate_train(videos):
    return build_clip(videos)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--feats', default='feats')
    ap.add_argument('--train_kind', default='gt', choices=['gt', 'det'])
    ap.add_argument('--dev_kind', default='gt', choices=['gt', 'det'])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=4, help='số video mỗi bước')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--n_spatial', type=int, default=1)
    ap.add_argument('--n_temporal', type=int, default=3)
    ap.add_argument('--no_focal', action='store_true')
    ap.add_argument('--no_pos_weight', action='store_true')
    ap.add_argument('--max_videos', type=int, default=0)
    ap.add_argument('--dev_videos', type=int, default=150)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--yolo_weights', default='', help='chỉ để ghi vào config')
    ap.add_argument('--imgsz', type=int, default=512, help='chỉ để ghi vào config')
    ap.add_argument('--out', default='runs/sttran')
    ap.add_argument('--resume', action='store_true', help='tiếp tục từ <out>/last.pth (model+optimizer+scheduler+epoch)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    index = pickle.load(open(a.index, 'rb'))
    ds = VideoFeatDataset(index, 'train', a.feats, kind=a.train_kind, limit=a.max_videos)
    D = torch.load(os.path.join(ds.dir, ds.vids[0] + '.pt'), weights_only=False)['D']
    write(f'train: {len(ds)} video, D={D}')
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate_train, num_workers=a.workers, drop_last=False)
    dev_dl = None
    try:
        dev_dl = make_loader(index, 'dev', a.feats, a.dev_kind, bs_videos=4, workers=a.workers, limit=a.dev_videos)
        write(f'dev: {len(dev_dl.dataset)} video ({a.dev_kind})')
    except FileNotFoundError:
        write('không có đặc trưng dev -> bỏ qua đo dev (chạy extract_feats --split dev --kind gt --max_videos 150)')

    cfg_model = dict(feat_dim=D, d=a.d, window=a.window, n_spatial=a.n_spatial, n_temporal=a.n_temporal)
    model = STTran(**cfg_model).to(a.device)
    pw_spa = pw_con = None
    if not a.no_pos_weight:
        spa_c, con_c, n_inst = label_counts(index, 'train')
        pw_spa = pos_weight_from_counts(spa_c, n_inst).to(a.device); pw_con = pos_weight_from_counts(con_c, n_inst).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = max(a.epochs * len(dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))

    log_path = os.path.join(a.out, 'train_log.csv')
    log_f = open(log_path, 'a' if a.resume else 'w', newline='', encoding='utf-8')
    log_w = None
    cfg = dict(args=dict(vars(a), bs=a.bs, epochs=a.epochs, lr_backbone=0.0, amp=False, ema=False, short=a.imgsz,
                         max_side=a.imgsz, frames_per_video=0, rfs=False, pos_weight=not a.no_pos_weight,
                         focal=not a.no_focal, detr_ckpt=a.yolo_weights or 'YOLO (extract_feats)'),
               n_params=sum(p.numel() for p in model.parameters()), model=cfg_model,
               run_info=run_info({'index': a.index, 'yolo_weights': a.yolo_weights}))
    t0, step, best, seen = time.time(), 0, -1.0, 0

    def save(name, ep):
        tmp = os.path.join(a.out, name + '.tmp')
        torch.save(dict(model=model.state_dict(), cfg=cfg_model, epoch=ep, step=step,
                        opt=opt.state_dict(), sched=sched.state_dict(), best=best, seen=seen), tmp)
        os.replace(tmp, os.path.join(a.out, name))

    start_ep = 0
    last_path = os.path.join(a.out, 'last.pth')
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=a.device, weights_only=False)
        model.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
            best, seen = ck.get('best', -1.0), ck.get('seen', 0)
        else:
            write('[cảnh báo] last.pth cũ không có optimizer/scheduler -> chỉ nạp trọng số')
        start_ep, step = ck['epoch'] + 1, ck['step']
        write(f'RESUME: tiếp tục từ epoch {start_ep + 1}/{a.epochs} (step {step}, best R@20={best:.1f})')
    elif a.resume:
        write('chưa có last.pth -> train từ đầu')

    for ep in range(start_ep, a.epochs):
        model.train()
        pbar = tqdm(dl, desc=f'epoch {ep + 1}/{a.epochs}', unit='bước')
        for clip in pbar:
            clip = clip.to(a.device)
            out = model(clip)
            loss, logs = relation_loss(out, clip, pw_spa, pw_con, focal=not a.no_focal)
            if not loss.requires_grad:   # lô không có token nào (video không có người/vật): không có gì để học -> bỏ qua
                continue
            if not torch.isfinite(loss):
                write(f'[cảnh báo] loss không hữu hạn ở bước {step}, bỏ qua'); opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; seen += clip.n_frames
            if step % 20 == 0:
                row = dict(step=step, epoch=ep, lr=opt.param_groups[0]['lr'], sec=round(time.time() - t0, 1),
                           total=float(loss), **logs)
                if log_w is None:
                    log_w = csv.DictWriter(log_f, fieldnames=list(row))
                    if os.path.getsize(log_path) == 0:
                        log_w.writeheader()
                log_w.writerow(row); log_f.flush()
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f'{float(loss):.2f}')
        if dev_dl is not None:
            r, _, _ = evaluate(Adapter(model), dev_dl, a.device, modes=('with',), ious=(0.5,), extra=False)
            r = r['with']['iou0.5']
            print(f'[dev] epoch {ep}: ' + ' '.join(f"R@{k}={r[f'R@{k}']:.1f}" for k in KS) + ' ' +
                  ' '.join(f"mR@{k}={r[f'mR@{k}']:.1f}" for k in KS))
            p = os.path.join(a.out, 'dev_log.csv'); new = not os.path.exists(p) or ep == 0
            with open(p, 'w' if ep == 0 else 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['epoch', 'step'] + [f'R@{k}' for k in KS] + [f'mR@{k}' for k in KS])
                w.writerow([ep, step] + [r[f'R@{k}'] for k in KS] + [r[f'mR@{k}'] for k in KS])
            if r['R@20'] > best:
                best = r['R@20']; save('best.pth', ep)
        else:
            save('best.pth', ep)
        save('last.pth', ep)
        cfg.update(epochs_done=ep + 1, steps=step, images_seen=seen, train_seconds=round(time.time() - t0, 1))
        json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=1, ensure_ascii=False)
    print('XONG. checkpoint:', os.path.join(a.out, 'best.pth'))


if __name__ == '__main__':
    main()
