"""
SellStein KYC — passive presentation-attack-detection (PAD) microservice.

Runs the Silent-Face MiniFASNet-V2 anti-spoof model (Apache-2.0) that CANNOT run
on Cloudflare Workers (workerd blocks runtime WASM instantiation — see the repo's
`reference_no_onnx_on_workers` note). This container is the server-TRUSTED PAD
layer: given a raw selfie frame it detects the face (YuNet), crops 2.7x the bbox,
and classifies [live, print-attack, replay-attack]. The operator-api Worker calls
POST /pad and folds the result into the KYC composite (vetoes the liveness score),
defense-in-depth alongside the existing vision-LLM liveness check.

NOT facial recognition: this only judges whether a presented face is LIVE vs a
photo/screen. It never identifies or matches a person.
"""
import base64
import os

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PAD_MODEL = os.path.join(MODEL_DIR, "minifasnet_v2.onnx")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")

# MiniFASNet-V2 (2.7_80x80): input (1,3,80,80) float32, BGR, /255, NCHW.
# The crop fed to the net is 2.7x the detected face bbox, centred on it.
INPUT_SIZE = 80
CROP_SCALE = 2.7
# A genuine live capture should clear this; below it the frame is treated as a
# possible spoof and the Worker routes the verification to manual review.
DEFAULT_FACE_SCORE_THRESHOLD = 0.6

_session = ort.InferenceSession(PAD_MODEL, providers=["CPUExecutionProvider"])
_in_name = _session.get_inputs()[0].name

# YuNet is a ~340KB ONNX face detector built into OpenCV. Input size is set
# per-image before each detect() call.
_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)

app = FastAPI(title="sellstein-kyc-pad", version="1.0.0")


class PadRequest(BaseModel):
    image_b64: str


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _detect_largest_face(img: np.ndarray):
    """Return the largest detected face bbox (x, y, w, h) or None."""
    h, w = img.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
    f = faces[0]
    return float(f[0]), float(f[1]), float(f[2]), float(f[3])


def _crop_scaled(img: np.ndarray, bbox, scale: float) -> np.ndarray:
    """Square crop of `scale`x the bbox, centred on the bbox, clamped to image."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * scale
    x1 = int(max(0, cx - side / 2.0))
    y1 = int(max(0, cy - side / 2.0))
    x2 = int(min(w, cx + side / 2.0))
    y2 = int(min(h, cy + side / 2.0))
    return img[y1:y2, x1:x2]


@app.get("/health")
def health():
    return {"ok": True, "model": "minifasnet_v2", "detector": "yunet", "input": INPUT_SIZE}


@app.post("/pad")
def pad(req: PadRequest):
    try:
        raw = base64.b64decode(req.image_b64, validate=False)
    except Exception:
        return {"ok": False, "error": "bad_base64"}
    if not raw:
        return {"ok": False, "error": "empty"}
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # decodes to BGR
    if img is None:
        return {"ok": False, "error": "decode_failed"}

    bbox = _detect_largest_face(img)
    if bbox is None:
        # No face is itself a strong spoof/quality signal — report it, let the
        # Worker decide (it has the other stages + the LLM to corroborate).
        return {"ok": True, "face_found": False, "live": 0.0, "print": 0.0, "replay": 0.0, "label": "no_face"}

    crop = _crop_scaled(img, bbox, CROP_SCALE)
    if crop.size == 0:
        return {"ok": True, "face_found": False, "live": 0.0, "print": 0.0, "replay": 0.0, "label": "no_face"}

    face = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0  # BGR [0,1]
    blob = np.transpose(face, (2, 0, 1))[np.newaxis, ...]  # NCHW (1,3,80,80)
    logits = _session.run(None, {_in_name: blob})[0][0]
    p = _softmax(np.asarray(logits, dtype=np.float64))
    live, printed, replay = float(p[0]), float(p[1]), float(p[2])
    label = ["live", "print", "replay"][int(np.argmax(p))]
    return {
        "ok": True,
        "face_found": True,
        "live": live,
        "print": printed,
        "replay": replay,
        "label": label,
        "is_live": bool(label == "live" and live >= DEFAULT_FACE_SCORE_THRESHOLD),
    }
