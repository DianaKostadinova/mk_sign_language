
"""
Live webcam demo for the DTW template matcher.

Instead of classifying a rolling window every frame, this segments signs
by hand motion:

  1. Hand appears and starts moving  -> start recording the segment
  2. Hand goes still (or disappears) -> segment ends
  3. The segment is resampled and matched against the reference
     templates with DTW; the top-3 closest letters are shown.

Run extract_templates.py first to build data/landmarks/dtw_templates.npz.
Press Q to quit.
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))
from alphabet.dtw_common import build_frame, classify

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
HAND_MODEL_PATH = ROOT / "data" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "data" / "pose_landmarker.task"
TEMPLATES_PATH  = ROOT / "data" / "landmarks" / "dtw_templates.npz"
USER_TEMPLATES  = ROOT / "data" / "landmarks" / "user_templates.npz"

# ── Segmentation tuning ───────────────────────────────────────────────────────
# Speed = wrist movement per frame in normalized image coords, EMA-smoothed.
START_SPEED    = 0.015   # moving faster than this starts a segment
END_SPEED      = 0.008   # slower than this counts as "still"
END_HOLD       = 6       # consecutive still frames that end a segment
PRE_ROLL       = 5       # frames kept from just before motion started
MIN_SEG_FRAMES = 10      # shorter segments are discarded as noise
MAX_SEG_FRAMES = 150     # safety cap — classify and reset
EMA_ALPHA      = 0.4

# ── Matching tuning ───────────────────────────────────────────────────────────
MARGIN_OK    = 1.10     # 2nd-best class must be 10%+ farther to count as confident
RESULT_HOLD  = 4.0      # seconds the last prediction stays on screen


# ── MediaPipe detectors (VIDEO mode — uses tracking, much faster per frame) ──
def make_hand_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_pose_detector():
    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Drawing helpers ───────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int):
    if size not in _FONT_CACHE:
        font = None
        for path in ["arial.ttf", "C:/Windows/Fonts/arial.ttf",
                     "C:/Windows/Fonts/arialuni.ttf", "C:/Windows/Fonts/seguiui.ttf"]:
            try:
                font = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                continue
        _FONT_CACHE[size] = font or ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_cyrillic(frame, texts):
    """Draw several (text, pos, size, color) items in one PIL round-trip."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    for text, pos, size, color in texts:
        draw.text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, pts):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


# ── Segment recorder ──────────────────────────────────────────────────────────
class SignSegmenter:
    """Speed-based start/stop detection for one sign performance."""

    def __init__(self):
        self.pre_roll   = deque(maxlen=PRE_ROLL)
        self.segment    = []
        self.recording  = False
        self.still_run  = 0
        self.ema_speed  = 0.0
        self.prev_wrist = None

    def _update_speed(self, wrist_xy: np.ndarray) -> float:
        if self.prev_wrist is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(wrist_xy - self.prev_wrist))
        self.prev_wrist = wrist_xy
        self.ema_speed  = EMA_ALPHA * speed + (1 - EMA_ALPHA) * self.ema_speed
        return self.ema_speed

    def _finish(self) -> np.ndarray | None:
        seg = self.segment
        self.segment, self.recording, self.still_run = [], False, 0
        if len(seg) >= MIN_SEG_FRAMES:
            return np.array(seg, dtype=np.float32)
        return None

    def update(self, feature: np.ndarray | None,
               wrist_xy: np.ndarray | None) -> np.ndarray | None:
        """
        Feed one frame. feature/wrist_xy are None when the hand wasn't detected.
        Returns a completed segment (T, POS_DIM) when one just ended, else None.
        """
        if feature is None:
            self.prev_wrist = None
            self.ema_speed  = 0.0
            self.pre_roll.clear()
            return self._finish() if self.recording else None

        speed = self._update_speed(wrist_xy)

        if not self.recording:
            self.pre_roll.append(feature)
            if speed > START_SPEED:
                self.segment   = list(self.pre_roll)
                self.recording = True
                self.still_run = 0
            return None

        self.segment.append(feature)

        if speed < END_SPEED:
            self.still_run += 1
        else:
            self.still_run = 0

        if self.still_run >= END_HOLD:
            # Drop most of the trailing still frames — they're the hold, not the sign
            if END_HOLD > 2:
                self.segment = self.segment[:-(END_HOLD - 2)]
            return self._finish()

        if len(self.segment) >= MAX_SEG_FRAMES:
            return self._finish()

        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    if not TEMPLATES_PATH.exists():
        print(f"Templates not found: {TEMPLATES_PATH}")
        print("Run extract_templates.py first.")
        return

    # Once you've recorded your own takes, match against ONLY those — the
    # reference videos are a different signer and don't transfer to a webcam.
    if USER_TEMPLATES.exists():
        user      = np.load(USER_TEMPLATES, allow_pickle=True)
        templates = user["templates"]
        labels    = user["labels"]
        print(f"Matching against YOUR {len(labels)} recorded takes "
              f"({len(set(labels))} letters). Reference videos ignored.")
    else:
        data      = np.load(TEMPLATES_PATH, allow_pickle=True)
        templates = data["templates"]
        labels    = data["labels"]
        print(f"Matching against {len(templates)} REFERENCE templates "
              f"(different signer — expect poor webcam accuracy).")

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()
    segmenter     = SignSegmenter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam open. Perform a sign, then hold still to classify. Press Q to quit.\n")

    last_ranked, last_margin, last_time = None, 0.0, 0.0
    timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        timestamp_ms += 33

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

        feature, wrist_xy = build_frame(hand_result, pose_result)
        hand_pts = []
        if feature is not None:
            h, w = frame.shape[:2]
            hand_pts = [[(int(lm.x * w), int(lm.y * h)) for lm in hand]
                        for hand in hand_result.hand_landmarks[:2]]

        segment = segmenter.update(feature, wrist_xy)
        if segment is not None:
            t0 = time.time()
            ranked, margin = classify(segment, templates, labels)
            last_ranked, last_margin, last_time = ranked, margin, time.time()
            top_str = "  ".join(f"{l}={d:.2f}" for l, d in ranked)
            print(f"Segment ({len(segment)} frames, {time.time() - t0:.2f}s) → "
                  f"{top_str}  margin={margin:.2f}")

        # ── Overlay ──────────────────────────────────────────────────────────
        for pts in hand_pts:
            draw_skeleton(frame, pts)

        # ASCII status via cv2 (fast); Cyrillic results via one PIL pass
        if segmenter.recording:
            status, status_color = f"* recording ({len(segmenter.segment)})", (80, 80, 255)
        elif feature is not None:
            status, status_color = "ready - start signing", (200, 200, 200)
        else:
            status, status_color = "no hand detected", (0, 0, 255)
        cv2.putText(frame, status, (30, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if last_ranked is not None and time.time() - last_time < RESULT_HOLD:
            confident = last_margin >= MARGIN_OK
            color  = (0, 255, 0) if confident else (0, 160, 255)
            runner = "   ".join(f"{l} {d:.2f}" for l, d in last_ranked[1:])
            texts  = [
                (last_ranked[0][0], (30, 15), 80, color),
                (f"margin {last_margin:.2f}" + ("" if confident else "  (uncertain)"),
                 (30, 110), 24, (255, 255, 255)),
                (runner, (30, 140), 22, (160, 160, 160)),
            ]
            frame = draw_cyrillic(frame, texts)

        cv2.imshow("MK Sign Language — Alphabet (DTW)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
