#!/usr/bin/env bash
# run_fast.sh — chạy trọn pipeline baseline 2 trong ~90 phút, không sửa gói agsgg/.
# Cách dùng (từ thư mục CHỨA gói agsgg/):
#   FRAMES=/path/to/frames INDEX=ag_index.pkl bash run_fast.sh
# Tuỳ chọn: DETR_CKPT=detr-r50.pth  BUDGET_MIN=90  OUT=runs/fast
set -euo pipefail
cd "$(dirname "$0")"
FRAMES="${FRAMES:?Đặt FRAMES=thư mục chứa frame (<FRAMES>/<video.mp4>/<000123.png>)}"
INDEX="${INDEX:-ag_index.pkl}"
DETR_CKPT="${DETR_CKPT:-detr-r50.pth}"
BUDGET_MIN="${BUDGET_MIN:-90}"
OUT="${OUT:-runs/fast}"

if [ ! -f "$DETR_CKPT" ]; then
  echo ">> tải pretrained DETR R50 (~160MB)..."
  wget -c https://dl.fbaipublicfiles.com/detr/detr-r50.pth -O "$DETR_CKPT"
fi

echo ">> 1/4 smoke test..."
python -m agsgg.tests.smoke_test
python -m agsgg.train --synthetic --epochs 1 --bs 2 --device cuda --amp

echo ">> 2/4 benchmark + chọn config vừa ${BUDGET_MIN} phút..."
ARGS=$(python fit_budget.py --index "$INDEX" --frames "$FRAMES" --detr "$DETR_CKPT" \
        --budget "$BUDGET_MIN" | tee /dev/stderr | grep '^RECOMMENDED' | sed 's/^RECOMMENDED //')
EPOCHS=$(echo "$ARGS" | grep -oP '(?<=--epochs )\d+')
echo ">> config: $ARGS"

echo ">> 3/4 train..."
python -m agsgg.train --index "$INDEX" --frames "$FRAMES" --detr_ckpt "$DETR_CKPT" \
  --amp --ema --rfs --pos_weight --focal --dev_n 200 --workers 8 \
  $ARGS --out "$OUT"

echo ">> 4/4 đánh giá test (checkpoint epoch cuối) + sinh số liệu báo cáo..."
python -m agsgg.evaluate --ckpt "$OUT/epoch$((EPOCHS-1)).pth" --index "$INDEX" \
  --frames "$FRAMES" --split test --out results/test_full --save_preds --bs 16 --workers 8
python -m agsgg.tools.make_report_assets --res results/test_full --train "$OUT" --out report_assets
echo ">> XONG. Số liệu: results/test_full/result.json, report_assets/, $OUT/{train_log.csv,dev_log.csv,config.json}"
