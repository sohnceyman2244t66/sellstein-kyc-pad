# SellStein KYC PAD service — MiniFASNet-V2 anti-spoof + YuNet face detect.
FROM python:3.11-slim

WORKDIR /app

# OpenCV (headless) still needs libGL + glib at runtime; curl pulls the models.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the models into the image so the container has no runtime network dep.
#  - MiniFASNet-V2 anti-spoof (Apache-2.0), ~1.7MB
#  - YuNet face detector (OpenCV Zoo), ~340KB
RUN mkdir -p /models \
    && curl -fsSL -o /models/minifasnet_v2.onnx \
        "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx" \
    && curl -fsSL -o /models/face_detection_yunet.onnx \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

COPY server.py .

ENV MODEL_DIR=/models
EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
