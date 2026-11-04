#!/usr/bin/env python
"""fit_budget.py — đo tốc độ GPU 60-90s rồi in ra config train vừa khít ngân sách thờigiangian.
Chạy từ thư mục CHỨA gói agsgg/:
    python fit_budget.py --index ag_index.pkl --frames <thu_muc_frame> --detr detr-r50.pth --budget 90
In dòng:  RECOMMENDED --epochs E --bs B --frames_per_video F
"""
import argparse, pickle, time
import torch
from torch.utils.data import DataLoader

def _f32(o):
    if isinstance(o, dict):
        return {k: _f32(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_f32(v) for v in o]
    return o.float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='ag_index.pkl')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--detr', default='detr-r50.pth')
    ap.add_argument('--budget', type=float, default=90, help='phút')
    ap.add_argument('--bs', type=int, default=0, help='0 = tự chọn theo VRAM')
    ap.add_argument('--short', type=int, default=384)
    ap.add_argument('--max_side', type=int, default=640)
    ap.add_argument('--dev_n', type=int, default=200)
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'cần GPU'
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'

    index = pickle.load(open(a.index, 'rb'))
    from agsgg.model import AGSGG, load_detr_weights
    from agsgg.criterion import SetCriterion
    from agsgg.data import collate_fn
    from agsgg.ag_dataset import AGDataset

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs = a.bs or (16 if vram >= 11 else 8)
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM {vram:.0f}GB -> bs={bs}')

    model = AGSGG(backbone_pretrained=False).to(dev)
    n_loaded = load_detr_weights(model, a.detr) if a.detr else 0
    print(f'đã nạp {n_loaded} tensor từ DETR')
    crit = SetCriterion().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # --- benchmark train (có backward) ---
    ds = AGDataset(index, 'train', a.frames, train=True, short=a.short,
                   max_side=a.max_side, frames_per_video=4, rfs=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                    drop_last=True, num_workers=8)
    model.train(); it = iter(dl)
    for _ in range(5):  # warmup
        try: x, mask, tgts = next(it)
        except StopIteration: it = iter(dl); x, mask, tgts = next(it)
        x, mask = x.to(dev), mask.to(dev)
        tgts = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _ = crit(_f32(model(x, mask)), tgts)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); t0 = time.time(); n = 0
    while time.time() - t0 < 40:
        try: x, mask, tgts = next(it)
        except StopIteration: it = iter(dl); x, mask, tgts = next(it)
        x, mask = x.to(dev), mask.to(dev)
        tgts = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in tgts]
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _ = crit(_f32(model(x, mask)), tgts)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        n += 1
    torch.cuda.synchronize()
    tps = n / (time.time() - t0)
    print(f'train: {tps:.1f} it/s  (batch {bs})')

    # --- benchmark eval (forward only, fp32 như evaluate.py) ---
    dev_ds = AGDataset(index, 'dev', a.frames, train=False, short=a.short,
                       max_side=a.max_side, limit=200)
    dev_dl = DataLoader(dev_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=8)
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for x, mask, _ in dev_dl:
            x, mask = x.to(dev), mask.to(dev)
            model(x, mask); n += 1
            if time.time() - t0 > 20: break
    torch.cuda.synchronize()
    eps_eval = n / (time.time() - t0)          # batch/s khi eval
    ips_eval = eps_eval * bs                   # ảnh/s khi eval
    print(f'eval:  {ips_eval:.0f} ảnh/s (fp32)')

    # --- ước lượng kích thước ---
    n_vid = len(index['train'])
    total_per_vid = [len(v) for v in index['train'].values()]
    test_frames = sum(len(v) for v in index['test'].values()) * 0.92  # bỏ frame không có person
    dev_time = a.dev_n / ips_eval / 60
    budget_s = a.budget * 60 - 240  # buffer 4 phút

    best = None
    for epochs in (6, 5, 4, 3, 2):
        for fpv in (8, 6, 4, 3, 2):
            fpe = sum(min(c, fpv) for c in total_per_vid)
            train_s = epochs * (fpe / bs / tps + dev_time * 60)
            test_s = test_frames / bs / eps_eval
            total = train_s + test_s
            if total <= budget_s:
                best = (epochs, fpv, total / 60, fpe)
                break
        if best: break
    if not best:  # chấp nhận vượt budget nhỏ nhất
        cand = [(e, f, sum(min(c, f) for c in total_per_vid)) for e in (2, 3) for f in (2, 1)]
        e, f, fpe = min(cand, key=lambda t: t[2])
        total = (e * (fpe / bs / tps + dev_time * 60) + test_frames / bs / eps_eval) / 60
        best = (e, f, total, fpe)
        print(f'!! cấu hình tối thiểu vẫn ~{total:.0f} phút — cân nhắc giảm test (evaluate --limit)')

    e, f, mins, fpe = best
    print(f'chọn: {e} epochs x fpv={f} (~{fpe/1000:.0f}k frame/epoch, '
          f'{fpe/bs:.0f} steps/epoch) -> ~{mins:.0f} phút (gồm eval test)')
    print(f'RECOMMENDED --epochs {e} --bs {bs} --frames_per_video {f}')

if __name__ == '__main__':
    main()
