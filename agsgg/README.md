# agsgg — RelTR-inspired SGG cho Action Genome

## Thứ tự chạy (từ thư mục CHỨA gói agsgg/)
    pip install torch torchvision scipy numpy pillow opencv-python tqdm matplotlib
    # 0. kiểm tra pipeline (dữ liệu giả)
    python -m agsgg.tests.smoke_test
    python -m agsgg.train --synthetic --epochs 1 --bs 2 --device cpu
    # 1. phân tích nhãn (tùy chọn) + lập chỉ mục
    python -m agsgg.tools.analyze_data --ann <thư mục có 2 file pkl>
    python -m agsgg.tools.build_index  --ann <thư mục có 2 file pkl> --out ag_index.pkl --protocol stage1 [--exclude dup.txt]
    # 2. cắt frame (dùng tools/dump_frames.py của repo ActionGenome, hoặc:)
    python -m agsgg.tools.extract_frames --index ag_index.pkl --videos <Charades_v1_480> --out <frames>
    # 3. kiểm tra frame khớp nhãn (xem ảnh trong check_out/)
    python -m agsgg.tools.check_frames --index ag_index.pkl --frames <frames> --draw 8
    # 4. train (lưu train_log.csv, dev_log.csv, config.json trong --out)
    python -m agsgg.train --index ag_index.pkl --frames <frames> --detr_ckpt detr-r50.pth \
        --epochs 6 --bs 4 --amp --ema --frames_per_video 8 --rfs --pos_weight --focal --out runs/agsgg
    # 5. đánh giá test (chạy MỘT lần, checkpoint epoch cuối) + lưu mọi số liệu cho báo cáo
    python -m agsgg.evaluate --ckpt runs/agsgg/epoch5.pth --index ag_index.pkl --frames <frames> --split test --out results/test_full --save_preds
    #    AG-filtersmall (xấp xỉ): python -m agsgg.tools.build_index --ann <...> --out ag_index_small.pkl --filter_small  rồi evaluate với --index ag_index_small.pkl --out results/test_small
    # 6. sinh bảng + hình cho báo cáo, và ví dụ định tính
    python -m agsgg.tools.make_report_assets --res results/test_full --train runs/agsgg --out report_assets
    python -m agsgg.tools.visualize_preds --res results/test_full --index ag_index.pkl --frames <frames> --split test --n 6

Chạy thử nhanh trước khi train lâu: thêm `--limit 20` cho evaluate; train thử `--epochs 1 --frames_per_video 1 --dev_n 50`.

Kết quả lưu cho báo cáo: train_log.csv (loss), dev_log.csv (R@K dev mỗi epoch), config.json (tham số, môi trường, hash, thời gian, lượt ảnh),
result.json (R@K/mR@K With/No x IoU 0.5/0.3, theo nhóm/lớp kèm n, phân rã lỗi, trùng lặp top-20, AP50 thực thể), per_frame.csv, preds.pkl.

## Chưa được kiểm chứng
Các file mới (ag_dataset.py, evaluate.py, runinfo.py, tools/*) và các sửa trong train.py / criterion.py
mới chỉ qua kiểm tra cú pháp (môi trường viết không có PyTorch/dữ liệu). Chạy smoke test + 1 epoch nhỏ trước.
Evaluator là bản tự viết theo mô tả STTran; muốn so sánh số với bài báo cần đối chiếu protocol (lọc frame, mapping lớp).
