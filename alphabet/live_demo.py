"""
Live webcam demo — shows the predicted letter in real time.

Run from project root:
    python alphabet/live_demo.py

Press Q to quit.
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))
from utils import normalize_landmarks, landmarks_from_result

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
WEIGHTS    = ROOT / "models" / "mlp_letters.pt"
LABELS     = ROOT / "models" / "label_encoder.npy"

# ── Load model ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model():
    checkpoint   = torch.load(WEIGHTS, map_location="cpu")
    label_names  = np.load(LABELS, allow_pickle=True)
    model = MLP(
        checkpoint["input_dim"],
        checkpoint["hidden_dims"],
        checkpoint["n_classes"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_names


# ── MediaPipe detector (IMAGE mode for per-frame inference) ───────────────────
def make_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ── Inference on a single frame ───────────────────────────────────────────────
@torch.no_grad()
def predict(frame_bgr, detector, model, label_names):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_image)

    raw = landmarks_from_result(result)
    if raw is None:
        return None, 0.0, None

    normalized = normalize_landmarks(raw)
    tensor     = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    logits     = model(tensor)
    probs      = torch.softmax(logits, dim=1)[0]
    top_idx    = probs.argmax().item()
    confidence = probs[top_idx].item()
    label      = label_names[top_idx]

    # Also return all landmark points for drawing
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]

    return label, confidence, pts


# ── Draw skeleton on frame ────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]

# Try to find a system font that supports Cyrillic
def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arial.ttf", "arialuni.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "C:/Windows/Fonts/seguiui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def put_text_unicode(frame: np.ndarray, text: str, pos: tuple, size: int, color: tuple) -> np.ndarray:
    """Draw Unicode text on an OpenCV frame using Pillow."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _find_font(size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, pts):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("Loading model ...")
    model, label_names = load_model()
    detector = make_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam open. Press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural
        label, confidence, pts = predict(frame, detector, model, label_names)

        if label is not None and confidence > 0.2:
            draw_skeleton(frame, pts)
            frame = put_text_unicode(frame, label, (30, 20), 80, (0, 255, 0))
            bar_w = int(300 * confidence)
            cv2.rectangle(frame, (30, 110), (30 + bar_w, 130), (0, 255, 0), -1)
            frame = put_text_unicode(frame, f"{confidence:.0%}", (340, 112), 24, (255, 255, 255))
        elif pts is not None:
            draw_skeleton(frame, pts)
            frame = put_text_unicode(frame, "?", (30, 20), 80, (0, 100, 255))
        else:
            frame = put_text_unicode(frame, "No hand detected", (30, 30), 28, (0, 0, 255))

        cv2.imshow("MK Sign Language — Alphabet", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
