
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
from utils import (normalize_landmarks, landmarks_from_result,
                   normalize_arm_landmarks, pose_landmarks_from_result)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
WEIGHTS         = ROOT / "models" / "lstm_letters.pt"
LABELS          = ROOT / "models" / "label_encoder.npy"

# ── Load model ────────────────────────────────────────────────────────────────
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
    checkpoint  = torch.load(WEIGHTS, map_location="cpu")
    label_names = np.load(LABELS, allow_pickle=True)
    model = SignLSTM(
        checkpoint["input_dim"],
        checkpoint["lstm_hidden"],
        checkpoint["lstm_layers"],
        checkpoint["n_classes"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_names, checkpoint["seq_len"]


# ── MediaPipe detectors (IMAGE mode for per-frame inference) ──────────────────
def make_hand_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Inference on a rolling frame buffer ──────────────────────────────────────
@torch.no_grad()
def predict(seq_buffer: list, model, label_names, seq_len: int):
    """Run LSTM on the current frame buffer. Returns label, confidence."""
    if len(seq_buffer) < seq_len:
        return None, 0.0

    # Evenly sample seq_len frames from the buffer
    indices = np.linspace(0, len(seq_buffer) - 1, seq_len, dtype=int)
    seq     = np.array([seq_buffer[i] for i in indices], dtype=np.float32)
    tensor  = torch.from_numpy(seq).unsqueeze(0)   # (1, seq_len, 63)

    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    top    = probs.argmax().item()
    return label_names[top], probs[top].item()


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
    model, label_names, seq_len = load_model()
    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print(f"Webcam open. Hold a sign steady for ~{seq_len} frames. Press Q to quit.\n")

    # Rolling buffer — stores the last seq_len*2 detected landmark frames
    buffer: list = []
    BUFFER_MAX = seq_len * 2

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)
        hand_raw    = landmarks_from_result(hand_result)
        pose_raw    = pose_landmarks_from_result(pose_result)

        pts = None
        if hand_raw is not None and pose_raw is not None:
            hand_vec   = normalize_landmarks(hand_raw)       # (63,)
            arm_vec    = normalize_arm_landmarks(pose_raw)   # (18,)
            normalized = np.concatenate([hand_vec, arm_vec]) # (81,)
            buffer.append(normalized)
            if len(buffer) > BUFFER_MAX:
                buffer.pop(0)

            h, w = frame.shape[:2]
            pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_result.hand_landmarks[0]]
        else:
            # Hand or pose lost — slowly drain the buffer so prediction fades out
            if buffer:
                buffer.pop(0)

        label, confidence = predict(buffer, model, label_names, seq_len)

        if label is not None and confidence > 0.2:
            if pts:
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
