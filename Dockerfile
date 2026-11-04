FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài torch trước. cu126 = GPU NVIDIA. Nếu chạy CPU thì đổi "cu126" thành "cpu"
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    YOLO_W=/app/weights/yolo_ag4_best.pt \
    STT_W=/app/weights/sttran4_best.pth \
    IMGSZ=640

EXPOSE 7860
CMD ["python", "app.py"]
