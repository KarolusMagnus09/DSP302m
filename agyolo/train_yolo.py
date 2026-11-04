"""Bước 2: fine-tune YOLO (Ultralytics, mặc định YOLO26) trên 36 lớp Action Genome.

  python -m agyolo.train_yolo --data yolo_data/data.yaml --model yolo26s.pt --imgsz 512 --batch 8 --time 0.75

--time (GIỜ) > 0: Ultralytics tự co giãn số epoch cho vừa thời gian (ưu tiên hơn --epochs).
Trọng số tốt nhất: runs/yolo/ag/weights/best.pt
GPU 4GB (T2000): yolo26s, --imgsz 512 --batch 8 (OOM thì --batch 4 hoặc --model yolo26n.pt). GPU 16GB: yolo26m, --batch 32.
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='yolo_data/data.yaml')
    ap.add_argument('--model', default='yolo26s.pt', help='yolo26n/s/m/l/x.pt (tự tải), hoặc đường dẫn .pt')
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--time', type=float, default=0.75, help='giờ; 0 = dùng --epochs')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=15, help='dừng sớm nếu val không cải thiện sau n epoch')
    ap.add_argument('--device', default='0', help="'0' = GPU đầu tiên, 'cpu'")
    ap.add_argument('--project', default='runs/yolo')
    ap.add_argument('--name', default='ag')
    ap.add_argument('--no_amp', action='store_true', help='tắt fp16 (nếu loss NaN)')
    ap.add_argument('--optimizer', default='auto', help="auto | SGD | AdamW. 'auto' BỎ QUA --lr0/--momentum; muốn chỉnh lr phải đặt SGD hoặc AdamW")
    ap.add_argument('--lr0', type=float, default=None, help='lr ban đầu (chỉ có tác dụng khi --optimizer không phải auto)')
    ap.add_argument('--lrf', type=float, default=None, help='lr cuối = lr0 * lrf')
    ap.add_argument('--warmup_epochs', type=float, default=None, help='0 = bỏ warmup (nên dùng khi fine-tune tiếp từ best.pt)')
    ap.add_argument('--resume', action='store_true', help='tiếp tục lần train bị ngắt (--model = last.pt)')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    kw = dict(data=a.data, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
              project=os.path.abspath(a.project), name=a.name, exist_ok=True, amp=not a.no_amp,
              plots=False, patience=a.patience)
    for k in ('lr0', 'lrf', 'warmup_epochs'):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    if a.optimizer != 'auto':
        kw['optimizer'] = a.optimizer
    if a.resume:
        model.train(resume=True)
    else:
        if a.time > 0:
            kw['time'] = a.time
        else:
            kw['epochs'] = a.epochs
        model.train(**kw)
    best = os.path.join(os.path.abspath(a.project), a.name, 'weights', 'best.pt')
    print('XONG. trọng số:', best)


if __name__ == '__main__':
    main()
