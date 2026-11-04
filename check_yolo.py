"""Chẩn đoán: YOLO một mình có thấy vật nào (vd. cái ghế) không, ở ngưỡng conf rất thấp?
Chạy:  python check_yolo.py --src anh_hoac_video.mp4 [--sec 1.0] [--conf 0.03]
Bỏ qua STTran, select_dets và mọi bộ lọc của app -> biết lỗi nằm ở mô hình hay ở bộ lọc/ngưỡng."""
import argparse, os
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--sec', type=float, default=1.0, help='nếu là video: lấy frame ở giây này')
ap.add_argument('--conf', type=float, default=0.03)
ap.add_argument('--yolo', default=os.environ.get('YOLO_W', 'weights/yolo_ag4_best.pt'))
ap.add_argument('--imgsz', type=int, default=int(os.environ.get('IMGSZ', '640')))
a = ap.parse_args()

m = YOLO(a.yolo)
print('Các lớp mô hình biết:', dict(m.names), '\n')

img = cv2.imread(a.src)
if img is None:                                   # không phải ảnh -> thử như video
    cap = cv2.VideoCapture(a.src)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.sec * 1000)
    ok, img = cap.read()
    if not ok:                                    # video không tua được -> đọc tuần tự
        cap = cv2.VideoCapture(a.src)
        for _ in range(int(30 * a.sec) + 1):
            ok, img = cap.read()
    assert ok and img is not None, 'Không đọc được ảnh/frame từ ' + a.src

r = m.predict(img, imgsz=a.imgsz, conf=a.conf, iou=0.6, max_det=100, verbose=False, half=False)[0]
rows = sorted(((float(b[4]), m.names[int(b[5])], [round(float(v)) for v in b[:4]]) for b in r.boxes.data.cpu()), reverse=True)
print(f'{len(rows)} detection ở conf >= {a.conf}:')
for s, n, bb in rows:
    print(f'  {s:.2f}  {n:<18} {bb}')
cv2.imwrite('check_yolo_out.jpg', r.plot())
print('\nĐã lưu ảnh có box: check_yolo_out.jpg')
