# Mỗi thư mục / file làm gì (tóm tắt)

Để **chạy app** chỉ cần: `app.py`, `demo_core.py`, `weights/`, `agyolo/`, `agsgg/constants.py`.
Phần còn lại là code dùng khi train / đánh giá, chạy app không đụng tới.

```
agapp/
├─ app.py              Giao diện web (Gradio): tải video lên -> video có box + triplet + timeline.json
├─ demo_core.py        Phần xử lý chính: đọc video, YOLO phát hiện, STTran đoán quan hệ, vẽ, ghi video, xuất JSON
├─ check_yolo.py       Công cụ nhỏ: cho YOLO chạy riêng trên 1 ảnh/video để xem nó thấy vật gì (dùng khi debug)
├─ requirements.txt    Danh sách thư viện cần cài
├─ README.md           Cách cài và chạy
├─ README_LLM.md      Định dạng timeline.json cho phần LLM
├─ README_CAU_TRUC.md  File này
│
├─ weights/            Trọng số đã train (hai file đi cùng nhau, không dùng lẻ)
│   ├─ yolo_ag4_best.pt     YOLO26m: phát hiện người + 35 loại vật
│   └─ sttran4_best.pth     STTran: đoán quan hệ người–vật (train trên đặc trưng của đúng bản YOLO trên)
│
├─ agyolo/             Hướng B1 (YOLO + STTran) — hướng đang dùng trong app
│   ├─ extract_feats.py     Lấy đặc trưng từ YOLO; app dùng 3 hàm: roi_feats, union, select_dets
│   ├─ sttran.py            Định nghĩa mô hình STTran; app dùng STTran và build_clip
│   ├─ prep_yolo.py         (train) đổi nhãn Action Genome sang định dạng YOLO
│   ├─ train_yolo.py        (train) huấn luyện YOLO
│   ├─ train_sttran.py      (train) huấn luyện STTran
│   ├─ eval_sttran.py       (đánh giá) tính R@K, mR@K
│   └─ README.md            Thứ tự chạy khi train lại
│
└─ agsgg/              Hướng A1 (RelTR) dùng để so sánh trong báo cáo + hằng số dùng chung
    ├─ constants.py         Tên các lớp vật và các loại quan hệ — app dùng file này
    ├─ model.py, transformer.py, matcher.py, criterion.py   (train) mô hình RelTR và hàm loss
    ├─ ag_dataset.py, data.py, sampling.py                  (train) đọc dữ liệu Action Genome
    ├─ train.py, fit_budget.py, run_fast.sh                 (train) huấn luyện
    ├─ evaluate.py, postprocess.py, runinfo.py              (đánh giá) tính số liệu cho báo cáo
    ├─ progress.py                                          thanh tiến độ
    ├─ preview.webp                                         ảnh minh họa
    └─ README.md                                            thứ tự chạy khi train lại
```

Lưu ý: thư mục `tests/` và `tools/` (nhắc trong README của `agsgg`/`agyolo`) **không** có trong gói này,
và dữ liệu Action Genome (`ag_index.pkl`, `frames/`) cũng không kèm theo. Muốn train lại cần có thêm các thứ đó.
