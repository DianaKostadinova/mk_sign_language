"""
Live webcam demo for WORDS using DTW — no encoder.

`word_demo.py` uses the AUTSL encoder + embedding gallery. Both of those were
built when VEL_WEIGHT was 8.0, so they are stale against the current descriptor
(1.0) and would score near-noise. This script skips the encoder entirely and
matches the way the letter demo does: DTW nearest-neighbour against the word
templates, which WERE migrated to the current weighting.

Difference from the letter demo: a word is ONE continuous sign, so the take is
not split — the whole clip is matched as a single segment.

Honest expectation: the letters work well because the gallery is your own
recorded takes (same camera, same session). No such re-recording exists for
words — the gallery comes from the reference videos, a different recording
domain. Measured cross-domain letter accuracy is 26.4% vs 86.2% in-domain, so
expect words to feel substantially worse than the letters just did. That gap is
the whole reason the encoder exists; this script measures how big it is.

    python words/word_demo_dtw.py
    python words/word_demo_dtw.py --topk 10
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "alphabet"))

from alphabet.dtw_common import build_frame, classify
from alphabet.sequence_demo import (make_hand_detector, make_pose_detector,
                                    draw_cyrillic, draw_skeleton)

TEMPLATES_PATH = ROOT / "data" / "landmarks" / "word_templates.npz"
STOP_ABSENT_S  = 1.2      # shorter than fingerspelling: one word per take
MARGIN_OK      = 1.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5,
                    help="how many candidates to show (default 5)")
    ap.add_argument("--templates", type=Path, default=TEMPLATES_PATH)
    args = ap.parse_args()

    if not args.templates.exists():
        print(f"{args.templates} not found — run words/extract_word_templates.py first.")
        return

    d = np.load(args.templates, allow_pickle=True)
    templates, labels = d["templates"], d["labels"].astype(str)
    vocab = sorted(set(labels))
    print(f"Loaded {len(templates)} templates covering {len(vocab)} words.")
    print(f"Chance is {1/len(vocab):.1%}. Gallery is the REFERENCE recordings, "
          "not your own takes — expect this to be harder than the letters.\n")
    print("Raise your hand to start; sign ONE word; take your hand out and")
    print(f"leave it out ~{STOP_ABSENT_S:.1f}s to finish. Q quits, R resets.\n")

    hand_detector, pose_detector = make_hand_detector(), make_pose_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frames, frame_times = [], []
    recording, absent_secs, timestamp_ms = False, 0.0, 0
    ranked: list = []
    margin = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        frame = cv2.flip(frame, 1)
        timestamp_ms += 33

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

        feature, _ = build_frame(hand_result, pose_result)
        present = feature is not None

        if present:
            h, w = frame.shape[:2]
            for hand in hand_result.hand_landmarks[:2]:
                draw_skeleton(frame, [(int(lm.x * w), int(lm.y * h)) for lm in hand])

        # ── one word per take ────────────────────────────────────────────────
        if not recording:
            if present:
                recording, frames, frame_times, absent_secs = True, [feature], [now], 0.0
        else:
            frame_times.append(now)
            if present:
                frames.append(feature)
                absent_secs = 0.0
            else:
                absent_secs += now - frame_times[-2]
                if absent_secs >= STOP_ABSENT_S:
                    if len(frames) >= 2:
                        ranked, margin = classify(np.array(frames, np.float32),
                                                  templates, labels, top_k=args.topk)
                        print(f"[{len(frames)} frames]  margin {margin:.3f}")
                        for i, (lab, dist) in enumerate(ranked, 1):
                            print(f"   {i}. {lab}  (d={dist:.3f})")
                        print()
                    else:
                        ranked, margin = [], 0.0
                    recording, frames, frame_times, absent_secs = False, [], [], 0.0

        # ── overlay ──────────────────────────────────────────────────────────
        if recording:
            if present:
                status, color = f"* RECORDING  {len(frames)} frames", (80, 80, 255)
            else:
                status = f"hand out - finishing in {max(0.0, STOP_ABSENT_S-absent_secs):.1f}s"
                color  = (0, 180, 255)
        else:
            status, color = "raise hand to sign a word", (200, 200, 200)
        cv2.putText(frame, status, (30, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if ranked:
            texts = [(f"1. {ranked[0][0]}", (30, 15), 54,
                      (0, 255, 0) if margin >= MARGIN_OK else (0, 160, 255))]
            for i, (lab, _) in enumerate(ranked[1:], start=2):
                texts.append((f"{i}. {lab}", (30, 15 + 54 + (i - 2) * 30), 26,
                              (180, 180, 180)))
            texts.append((f"margin {margin:.2f}"
                          + ("" if margin >= MARGIN_OK else "  (uncertain)"),
                          (30, 15 + 54 + len(ranked) * 30), 22, (150, 150, 150)))
            frame = draw_cyrillic(frame, texts)

        cv2.imshow("MK Sign Language — Words (DTW)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            recording, frames, frame_times, absent_secs = False, [], [], 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
