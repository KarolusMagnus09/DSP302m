"""Vòng huấn luyện mẫu. Cần viết thêm adapter dữ liệu AG (build_dataset) cho định dạng nhãn của bạn.
Ví dụ chạy kiểm tra:  python -m agsgg.train --synthetic --epochs 1 --bs 2 --device cpu
"""
import argparse, copy, csv, json, math, os, pickle, time
import torch
from torch.utils.data import DataLoader

from .model import AGSGG, load_detr_weights
from .criterion import SetCriterion, pos_weight_from_counts
from .data import SyntheticDataset, collate_fn
from .progress import tqdm, write


class EMA:
    def __init__(self, model, decay=0.9998):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def _f32(o):
    if isinstance(o, dict):
        return {k: _f32(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_f32(v) for v in o]
    return o.float()


def build_dataset(args):
    if args.synthetic:
        return SyntheticDataset()
    from .ag_dataset import AGDataset
    index = pickle.load(open(args.index, 'rb'))
    return AGDataset(index, 'train', args.frames, train=True, short=args.short, max_side=args.max_side,
                     frames_per_video=args.frames_per_video, rfs=args.rfs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--detr_ckpt', default='')
    ap.add_argument('--out', default='runs/agsgg')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--frames', default='', help='thư mục frame: <frames>/<video.mp4>/<000123.png>')
    ap.add_argument('--short', type=int, default=384); ap.add_argument('--max_side', type=int, default=640)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--frames_per_video', type=int, default=0, help='>0: mỗi epoch lấy lại ngẫu nhiên k frame/video')
    ap.add_argument('--rfs', action='store_true', help='Repeat Factor Sampling (t=0.1, cap=10)')
    ap.add_argument('--pos_weight', action='store_true', help='BCE pos_weight theo tần suất lớp trên train')
    ap.add_argument('--focal', action='store_true', help='focal loss (gamma=2) cho spatial/contacting')
    ap.add_argument('--dev_n', type=int, default=300, help='>0: mỗi epoch đo R@K trên n frame dev (cố định) -> dev_log.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--time_budget', type=float, default=0,
                    help='>0: ngân sách thời gian train (PHÚT). Đo tốc độ ~60 bước đầu rồi tự đặt tổng số bước '
                         'sao cho xong đúng giờ (cosine LR chạy theo số bước này); --epochs bị bỏ qua')
    ap.add_argument('--eval_every', type=int, default=1500, help='(chế độ time_budget) cứ n bước: lưu last.pth + đo dev')
    ap.add_argument('--ema_decay', type=float, default=0.999, help='0.9998 hợp train dài; run ngắn dùng 0.999')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    small = args.synthetic  # mô hình nhỏ cho smoke run
    model = AGSGG(**(dict(d=64, nhead=4, enc_layers=1, dec_layers=2, rel_layers=1, dff=128, roi_size=2)
                     if small else dict(backbone_pretrained=not args.detr_ckpt))).to(args.device)
    if args.detr_ckpt:
        print('đã nạp', load_detr_weights(model, args.detr_ckpt), 'tensor từ DETR')
    kw = {}
    if args.focal:
        kw['rel_loss'] = 'focal'
    if args.pos_weight and not args.synthetic:
        from .ag_dataset import label_counts
        spa_c, con_c, n_inst = label_counts(pickle.load(open(args.index, 'rb')), 'train')
        kw['pw_spa'] = pos_weight_from_counts(spa_c, n_inst); kw['pw_con'] = pos_weight_from_counts(con_c, n_inst)
    crit = SetCriterion(**kw).to(args.device)
    ema = EMA(model, args.ema_decay) if args.ema else None

    ds = build_dataset(args)
    mk_dl = lambda: DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, drop_last=True,
                               num_workers=0 if args.synthetic else args.workers)
    dl = mk_dl()
    n_per_epoch = (ds.expected_len() if hasattr(ds, 'expected_len') else len(ds)) // args.bs
    groups = [{'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': args.lr},
              {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr_backbone}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    budget = args.time_budget > 0
    st = {'total': 10 ** 9 if budget else max(args.epochs * n_per_epoch, 1)}   # budget: đặt lại sau khi đo tốc độ
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / st['total'], 1.0)))))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and args.device.startswith('cuda'))

    dev_dl = None
    if args.dev_n > 0 and not args.synthetic:
        from .ag_dataset import AGDataset
        dev_ds = AGDataset(pickle.load(open(args.index, 'rb')), 'dev', args.frames, train=False, short=args.short,
                           max_side=args.max_side, limit=args.dev_n)
        dev_dl = DataLoader(dev_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn, num_workers=args.workers)

    from .runinfo import run_info
    cfg = dict(args=vars(args), n_params=sum(p.numel() for p in model.parameters()), steps_per_epoch_est=n_per_epoch,
               run_info=run_info({'index': None if args.synthetic else args.index, 'detr_ckpt': args.detr_ckpt}))
    save_cfg = lambda: json.dump(cfg, open(os.path.join(args.out, 'config.json'), 'w'), indent=1, ensure_ascii=False)
    log_f = open(os.path.join(args.out, 'train_log.csv'), 'w', newline='', encoding='utf-8')
    log_w = None
    t_start = time.time()

    def ckpt_and_dev(fname, ep):
        torch.save({'model': (ema.shadow if ema else model).state_dict(), 'epoch': ep, 'step': step},
                   os.path.join(args.out, fname))
        if dev_dl is not None:
            from .evaluate import evaluate, KS
            m_eval = ema.shadow if ema else model
            was = model.training
            r, _, _ = evaluate(m_eval, dev_dl, args.device, modes=('with',), ious=(0.5,), extra=False)
            model.train(was)
            r = r['with']['iou0.5']
            print(f'[dev{args.dev_n}] {fname} step {step}: ' + ' '.join(f"R@{k}={r[f'R@{k}']:.1f}" for k in KS))
            p = os.path.join(args.out, 'dev_log.csv')
            new = not os.path.exists(p)
            with open(p, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['epoch', 'step'] + [f'R@{k}' for k in KS] + [f'mR@{k}' for k in KS])
                w.writerow([ep, step] + [r[f'R@{k}'] for k in KS] + [r[f'mR@{k}'] for k in KS])

    step = 0
    done = False
    t_cal = None
    dev_sec = 0.0
    n_bad = 0
    for ep in range(10 ** 6 if budget else args.epochs):
        model.train()
        if hasattr(ds, 'resample') and ep > 0:
            ds.resample(); dl = mk_dl()
        pbar = tqdm(dl, desc=f'epoch {ep + 1}/{args.epochs}', unit='batch')
        for x, mask, tgts in pbar:
            x, mask = x.to(args.device), mask.to(args.device)
            tgts = [{k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
            with torch.autocast(args.device.split(':')[0], dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(x, mask)
            loss, logs = crit(_f32(out), tgts)
            if not torch.isfinite(loss):          # bỏ qua batch NaN/inf thay vì sập; sập hẳn nếu liên tục
                n_bad += 1
                if n_bad == 1:
                    bad = [k for k, v in out.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
                    write(f'[cảnh báo] loss không hữu hạn ở bước {step}: output NaN={bad}, ảnh NaN={not torch.isfinite(x).all().item()}'
                          f', vid={[t["vid"] for t in tgts if "vid" in t]}')
                if n_bad > 50:
                    raise RuntimeError('quá 50 batch NaN: bỏ --amp, hoặc hạ --lr 5e-5')
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update(); sched.step()
            if ema:
                ema.update(model)
            step += 1
            if budget:
                if step == 10:
                    t_cal = time.time()
                if step == 70 and t_cal is not None:     # đo tốc độ trên 60 bước (đã khởi động)
                    sps = (time.time() - t_cal) / 60
                    train_sec = args.time_budget * 60 - (time.time() - t_start)
                    n_eval = max(1, int(train_sec / sps / args.eval_every))
                    st['total'] = step + int(max(train_sec - n_eval * 45, 0.8 * train_sec) / sps)   # chừa ~45s/lần dev
                    print(f'[budget] {sps:.3f}s/bước -> tổng {st["total"]} bước (~{st["total"] * args.bs} ảnh) trong {args.time_budget} phút')
                    cfg['budget'] = dict(sec_per_step=sps, total_steps=st['total'])
                if step % args.eval_every == 0 and step < st['total']:
                    ckpt_and_dev('last.pth', ep)
                if step >= st['total']:
                    done = True
                    break
            if step % 20 == 0 or args.synthetic:
                row = dict(step=step, epoch=ep, lr=opt.param_groups[0]['lr'], sec=round(time.time() - t_start, 1),
                           **{k: float(v) for k, v in logs.items() if not k.startswith('aux')})
                if log_w is None:
                    log_w = csv.DictWriter(log_f, fieldnames=list(row)); log_w.writeheader()
                log_w.writerow(row); log_f.flush()
                write(f'ep {ep} step {step} ' + ' '.join(f'{k}={v:.3f}' for k, v in logs.items() if not k.startswith('aux')))
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{float(logs['total']):.2f}")
        if not budget:
            ckpt_and_dev(f'epoch{ep}.pth', ep)
        cfg.update(epochs_done=ep + 1, steps=step, images_seen=step * args.bs, train_seconds=round(time.time() - t_start, 1))
        save_cfg()
        if done:
            break
    if budget:
        ckpt_and_dev('final.pth', ep)
        cfg.update(steps=step, images_seen=step * args.bs, train_seconds=round(time.time() - t_start, 1))
        save_cfg()
        print('XONG. checkpoint cuối:', os.path.join(args.out, 'final.pth'))


if __name__ == '__main__':
    main()
