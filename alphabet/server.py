"""
FastAPI backend — streams webcam + sign predictions to the Angular frontend.

Run from project root:
    python alphabet/server.py

Endpoints:
    GET  /video_feed          MJPEG stream (use as <img src="...">)
    WS   /ws/predictions      JSON { letter, confidence } at ~20 fps
"""

import sys, asyncio, threading, json, time
import cv2, numpy as np, torch, torch.nn as nn
import mediapipe as mp
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from utils import (normalize_landmarks, landmarks_from_result,
                   normalize_arm_landmarks, pose_landmarks_from_result)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL = ROOT / "data" / "pose_landmarker.task"
WEIGHTS    = ROOT / "models" / "lstm_letters.pt"
LABELS     = ROOT / "models" / "label_encoder.npy"
SEQ_LEN    = 20


# ── Model ─────────────────────────────────────────────────────────────────────

class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden, layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def load_model():
    ckpt        = torch.load(WEIGHTS, map_location="cpu")
    label_names = np.load(LABELS, allow_pickle=True)
    model = SignLSTM(ckpt["input_dim"], ckpt["lstm_hidden"],
                     ckpt["lstm_layers"], ckpt["n_classes"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, label_names, ckpt["seq_len"]


@torch.no_grad()
def predict(seq_buffer, model, label_names, seq_len):
    if len(seq_buffer) < seq_len:
        return None, 0.0
    indices = np.linspace(0, len(seq_buffer) - 1, seq_len, dtype=int)
    seq     = np.array([seq_buffer[i] for i in indices], dtype=np.float32)
    tensor  = torch.from_numpy(seq).unsqueeze(0)
    logits  = model(tensor)
    probs   = torch.softmax(logits, dim=1)[0]
    top     = probs.argmax().item()
    return str(label_names[top]), float(probs[top])


# ── Shared state ──────────────────────────────────────────────────────────────

_lock         = threading.Lock()
_latest_jpeg  = None
_latest_pred  = {"letter": None, "confidence": 0.0}
_is_running   = False


# ── Detection thread ──────────────────────────────────────────────────────────

def detection_loop():
    global _latest_jpeg, _latest_pred, _is_running

    model, label_names, seq_len = load_model()

    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap    = cv2.VideoCapture(0)
    buf    = []
    BUF_MAX = seq_len * 2

    _is_running = True
    print("Detection loop started.")

    while _is_running:
        ok, frame = cap.read()
        if not ok:
            break

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_res  = hand_det.detect(mp_img)
        pose_res  = pose_det.detect(mp_img)
        hand_raw  = landmarks_from_result(hand_res)
        pose_raw  = pose_landmarks_from_result(pose_res)

        if hand_raw is not None and pose_raw is not None:
            vec = np.concatenate([
                normalize_landmarks(hand_raw),
                normalize_arm_landmarks(pose_raw)
            ])
            buf.append(vec)
            if len(buf) > BUF_MAX:
                buf.pop(0)

        letter, conf = predict(buf, model, label_names, seq_len)

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        with _lock:
            _latest_jpeg = jpg.tobytes()
            _latest_pred = {
                "letter":     letter,
                "confidence": round(conf, 3),
            }

    cap.release()
    _is_running = False
    print("Detection loop stopped.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()


def _mjpeg_frames():
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        time.sleep(1 / 30)   # ~30 fps cap


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/predictions")
async def predictions_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with _lock:
                pred = dict(_latest_pred)
            await ws.send_text(json.dumps(pred))
            await asyncio.sleep(0.05)   # 20 updates/s
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
