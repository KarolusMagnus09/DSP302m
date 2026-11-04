# Đầu ra cho phần LLM

Mô hình (YOLO26 + STTran) biến video thành `timeline.json`. Phần LLM chỉ cần đọc file này, không cần chạy mô hình.

## Tạo file
- Giao diện: `python app.py` -> tải video -> Chạy -> tải `timeline.json` ở ô "JSON cho LLM".
- Dòng lệnh:  `python demo_core.py --video input.mp4 --out demo_out.mp4 --json timeline.json`

## Định dạng
```json
{
  "video": "test2.mp4",
  "fps": 29.7,
  "duration_sec": 51.2,
  "params": {"conf": 0.25, "rel_thr": 0.3, "ctx_sec": 0.5},
  "objects_seen": {"closet/cabinet": 480, "door": 120, "chair": 35},
  "segments": [
    {"triplet": "person - holding - clothes", "start": 0.0, "end": 1.33, "frames": 40, "avg_score": 0.8}
  ]
}
```
- `segments`: mỗi phần tử là một quan hệ người–vật kéo dài từ `start` đến `end` (giây). Các lần xuất hiện cách nhau
  không quá 0.5 s được gộp thành một đoạn; đoạn ngắn hơn ~0.3 s bị bỏ vì thường là nhiễu.
- `avg_score` (0–1) = xác suất quan hệ x độ tin của box vật, trung bình trên đoạn. Càng thấp càng kém chắc.
- `objects_seen`: số lần mỗi vật được phát hiện (cộng dồn theo frame), cho biết cảnh có những vật nào.

## Lưu ý khi viết prompt cho LLM
- Dữ liệu CÓ NHIỄU (mô hình dự đoán, không phải sự thật). Nên dặn LLM: chỉ mô tả điều có trong `segments`,
  không bịa thêm chi tiết, nói "có thể" với các đoạn `avg_score` thấp.
- Chỉ có một người (`person`); chỉ có 35 lớp vật của Action Genome; vật ngoài danh sách sẽ không xuất hiện.
- Nên lọc `avg_score` >= ~0.2 và/hoặc `frames` đủ lớn trước khi đưa vào prompt để tiết kiệm token.
