# agyolo — YOLO26 (detector) + STTran (quan hệ) trên Action Genome

Giải thích nhanh: YOLO26 fine-tune trên 36 lớp AG (person + 35 vật) lo phần PHÁT HIỆN. STTran (viết gọn lại, chạy trên
đặc trưng YOLO) lo phần QUAN HỆ giữa người và từng vật, có nhìn ngữ cảnh theo thời gian (nhiều frame của cùng video).
Dùng lại `ag_index.pkl`, `frames/` và evaluator của `agsgg/` => số R@K / mR@K / phân rã lỗi so sánh trực tiếp được với baseline agsgg.

Đặt thư mục `agyolo/` CẠNH `agsgg/` (cùng cấp với ag_index.pkl và frames/). Cần: `pip install ultralytics` (tự kéo torch nếu chưa có).
Chạy mọi lệnh từ thư mục chứa `agyolo/` và `agsgg/`.

## Thứ tự chạy (Windows cmd, mỗi lệnh một dòng)
```
:: 1. đổi nhãn AG sang định dạng YOLO (không copy ảnh; train 20k frame, val 1.5k)
python -m agyolo.prep_yolo --index ag_index.pkl --frames frames --out yolo_data --max_train 20000 --val_n 1500

:: 2. fine-tune YOLO26 (--time: số GIỜ; tự co giãn epoch). T2000 4GB: yolo26s, batch 8; OOM thì --batch 4
python -m agyolo.train_yolo --data yolo_data/data.yaml --model yolo26s.pt --imgsz 512 --batch 8 --time 1.0 --workers 4

:: 3. trích đặc trưng (best.pt của YOLO). Train: một phần video (giữ nguyên chuỗi frame); dev/test: gt; test thêm det
python -m agyolo.extract_feats --weights runs/yolo/ag/weights/best.pt --index ag_index.pkl --frames frames --split train --kind gt --video_frac 0.2 --out feats
python -m agyolo.extract_feats --weights runs/yolo/ag/weights/best.pt --index ag_index.pkl --frames frames --split dev --kind gt --max_videos 100 --out feats
python -m agyolo.extract_feats --weights runs/yolo/ag/weights/best.pt --index ag_index.pkl --frames frames --split test --kind both --max_videos 150 --out feats

:: 4. train STTran (nhanh, vài phút/epoch)
python -m agyolo.train_sttran --index ag_index.pkl --feats feats --epochs 12 --bs 4 --out runs/sttran --yolo_weights runs/yolo/ag/weights/best.pt

:: 5. đánh giá: PredCls (box thật, đo riêng phần quan hệ) và SGDet (box YOLO, cả pipeline)
python -m agyolo.eval_sttran --ckpt runs/sttran/best.pth --index ag_index.pkl --feats feats --kind gt  --split test --out results/sttran_predcls --save_preds
python -m agyolo.eval_sttran --ckpt runs/sttran/best.pth --index ag_index.pkl --feats feats --kind det --split test --out results/sttran_sgdet --save_preds

:: 6. bảng + hình cho báo cáo (dùng lại công cụ của agsgg)
python -m agsgg.tools.make_report_assets --res results/sttran_sgdet --train runs/sttran --out report_assets_sttran
```
Mỗi bước chạy lại được (bước 3 bỏ qua video đã trích). Số liệu so với baseline agsgg: lấy SGDet (kind det) cho cùng tập test.

## Điều chỉnh theo GPU / thời gian
- GPU 16GB (T4/P100): `--model yolo26m.pt --batch 32 --workers 8`, `extract_feats --bs 32 --half`.
- Trích đặc trưng và train YOLO là hai bước tốn thời gian nhất; thu nhỏ bằng `--max_train`, `--video_frac`, `--max_videos`.
- STTran: `--train_kind det` để học thêm trên box YOLO (cần `extract_feats --split train --kind det/both`; chỉ tính loss trên vật khớp GT).
- Cửa sổ thời gian: `--window` (mặc định 2 như STTran). `--d 512 --n_temporal 3` ~ vài triệu tham số.

## Thiết kế (để viết báo cáo)
- Detector: YOLO26 (Ultralytics), 36 lớp, đã pretrain COCO rồi fine-tune AG. Frame không có bbox người bị bỏ khỏi train YOLO.
- Đặc trưng: hook vào đầu vào head Detect (P3/P4/P5), RoIAlign 3x3 -> trung bình -> nối 3 tầng; mỗi cặp có vector người / vật / union box.
- STTran: token cặp (người, vật) = [feat người, feat vật, feat union, conv trên mặt nạ 2x27x27, embedding lớp vật, hình học]
  -> Spatial encoder (attention trong frame) -> Temporal encoder (cửa sổ trượt giữa các frame, mã hoá vị trí frame) -> 3 đầu
  attention (softmax), spatial và contacting (sigmoid, focal + pos_weight).
- Khác STTran gốc: đặc trưng từ YOLO (không phải Faster R-CNN); cửa sổ trượt áp dụng ở mỗi lớp temporal; nhiều video ghép chuỗi với mặt nạ khối.
- Đánh giá: cùng evaluator agsgg (R@K/mR@K With/No constraint, IoU 0.5/0.3, phân rã lỗi, mAP50). SGDet: người = detection person điểm cao nhất, vật = top 12 (conf >= 0.1).

## Đã kiểm tra gì
Toàn bộ pipeline chạy hết từ đầu tới cuối trên dữ liệu giả (CPU): prep -> train YOLO26n -> trích đặc trưng -> train STTran -> eval PredCls/SGDet -> make_report_assets.
STTran học được quy tắc quan hệ giả (R@K dev 0 -> 100%). CHƯA chạy trên GPU và dữ liệu Action Genome thật; số liệu thật cần bạn chạy.
Windows: symlink có thể không tạo được -> dùng junction hoặc copy (prep_yolo tự thử). Chưa kiểm chứng trên Windows.
