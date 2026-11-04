# App demo: quan hệ người–vật trong video (YOLO26 + STTran)

## Cấu trúc thư mục (gói này đã đủ, không cần ghép thêm gì)
```
agapp/
  app.py  demo_core.py  requirements.txt  check_yolo.py
  agyolo/  agsgg/                      <- code đã dùng để train (STTran, hằng số, RoIAlign)
  weights/yolo_ag4_best.pt             <- YOLO26m, train imgsz=640
  weights/sttran4_best.pth             <- STTran, train trên đặc trưng của đúng bản YOLO này
  README_LLM.md                        <- định dạng timeline.json cho phần LLM
```
Mỗi thư mục/file làm gì: xem `README_CAU_TRUC.md`.
Không cần `ag_index.pkl`, `frames/`, hay `feats*` để chạy app (chỉ cần khi train / đánh giá).
Cần ffmpeg: `pip install imageio-ffmpeg` (đã có trong requirements.txt) là đủ, không cần cài ffmpeg hệ thống.

## Cài và chạy (Windows cmd, trong thư mục app)
```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python app.py
```
Mở http://127.0.0.1:7860, tải video lên, bấm Chạy. Lần đầu nạp model mất vài giây.

Chạy dòng lệnh không cần giao diện:
```
python demo_core.py --video input.mp4 --out demo_out.mp4 --max_sec 10
```

## Lưu ý
- `IMGSZ` phải bằng `imgsz` lúc train YOLO (640). Đặt biến môi trường `IMGSZ` nếu khác; đường dẫn trọng số đổi bằng `YOLO_W`, `STT_W`.
- Không có GPU thì tự chạy CPU (chậm hơn nhiều). GPU 4 GB (Quadro T2000): chạy suy luận fp16, chưa kiểm chứng trên máy bạn.
- Video trong trình duyệt cần H.264: `imageio-ffmpeg` trong requirements lo việc chuyển; nếu video kết quả không phát được thì cài ffmpeg hệ thống.
- Hai file `.pt` / `.pth` phải đi cùng nhau: STTran được train trên đặc trưng của đúng bản YOLO này.
