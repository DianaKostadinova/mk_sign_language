"""
Live webcam demo for the WORD recognizer — encoder + gallery, not DTW.

Segmentation reuses the exact same speed-based start/stop logic as
alphabet/dtw_demo.py (SignSegmenter). The difference is what happens once a
segment completes:

  DTW (letters):  segment -> full 21pt features -> template -> DTW vs every
                  reference template.
  Encoder (words): segment -> COARSE 10pt-per-hand features (build_frame_coarse,
                  matching what the AUTSL-trained encoder saw) -> template ->
                  embed with sign_encoder.pt -> cosine-nearest-neighbour
                  against the pre-embedded Macedonian gallery.

This is the first real test of cross-signer generalization on Macedonian
words: you are a different signer from whoever is in the reference word
videos, which the DTW self-retrieval test could never check.

While recording a segment, the recent wrist trajectory is drawn as a fading
trail — a direct visual of the motion actually being captured (not a DTW
cost matrix, since words no longer go through DTW at all).

Run word_templates_coarse.npz + word_embeddings.npz + models/sign_encoder.pt
must all already exist (see words/extract_word_templates_coarse.py and
notebooks/encoder_training.py build_macedonian_gallery()).

Press Q to quit.
"""
import sys
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "alphabet"))     # for `from dtw_common import ...` inside encoder_training.py
sys.path.append(str(ROOT / "notebooks"))

from alphabet.dtw_common import build_frame_coarse, make_template, COARSE_POS_DIM
from alphabet.dtw_demo import (                 # reuse, don't reinvent
    make_hand_detector, make_pose_detector, draw_skeleton, draw_cyrillic,
    SignSegmenter,
)
from encoder_training import SignEncoder, DEVICE

GALLERY_PATH  = ROOT / "data" / "landmarks" / "word_embeddings.npz"
ENCODER_PATH  = ROOT / "models" / "sign_encoder.pt"
TRAIL_LEN     = 40     # how many recent wrist points to draw as the motion trail
RESULT_HOLD   = 5.0    # seconds the last prediction stays on screen
TOP_K         = 5


def main():
    if not GALLERY_PATH.exists():
        print(f"Gallery not found: {GALLERY_PATH}")
        print("Run notebooks/encoder_training.py build_macedonian_gallery() first.")
        return
    if not ENCODER_PATH.exists():
        print(f"Encoder not found: {ENCODER_PATH}")
        return

    g = np.load(GALLERY_PATH, allow_pickle=True)
    gallery_emb = torch.tensor(g["embeddings"], dtype=torch.float32).to(DEVICE)
    gallery_lab = np.array([str(x) for x in g["labels"]])
    print(f"Loaded gallery: {len(gallery_lab)} embeddings, "
          f"{len(set(gallery_lab))} unique words.")

    in_dim = COARSE_POS_DIM * 2   # make_template appends velocity, doubling the per-frame dim

    encoder = SignEncoder(in_dim).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=True))
    encoder.eval()

    hand_detector = make_hand_detector()
    pose_detector = make_pose_detector()
    segmenter     = SignSegmenter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam open. Perform a word, then hold still to recognize. Press Q to quit.\n")

    trail = deque(maxlen=TRAIL_LEN)
    last_ranked, last_time = None, 0.0
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

        feature, wrist_xy = build_frame_coarse(hand_result, pose_result)
        hand_pts = []
        if feature is not None:
            h, w = frame.shape[:2]
            hand_pts = [[(int(lm.x * w), int(lm.y * h)) for lm in hand]
                        for hand in hand_result.hand_landmarks[:2]]
            trail.append((int(wrist_xy[0] * w), int(wrist_xy[1] * h)))
        else:
            trail.clear()

        segment = segmenter.update(feature, wrist_xy)
        if segment is not None:
            t0 = time.time()
            tmpl = make_template(segment)                          # (32, COARSE_POS_DIM*2)
            with torch.no_grad():
                q = encoder(torch.tensor(tmpl[None], dtype=torch.float32).to(DEVICE))
                sims = (q @ gallery_emb.t()).cpu().numpy()[0]
            order = sims.argsort()[::-1]
            # de-duplicate: same word can appear many times in the gallery (trim variants)
            seen, ranked = set(), []
            for i in order:
                lab = gallery_lab[i]
                if lab in seen:
                    continue
                seen.add(lab); ranked.append((lab, float(sims[i])))
                if len(ranked) >= TOP_K:
                    break
            last_ranked, last_time = ranked, time.time()
            top_str = "  ".join(f"{l}={s:.2f}" for l, s in ranked)
            print(f"Segment ({len(segment)} frames, {time.time() - t0:.2f}s) → {top_str}")

        # ── Overlay ──────────────────────────────────────────────────────────
        for pts in hand_pts:
            draw_skeleton(frame, pts)

        # Motion trail — fading polyline of recent wrist positions.
        pts = list(trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            color = (int(255 * alpha), int(80 + 100 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(4 * alpha)))

        if segmenter.recording:
            status, status_color = f"* recording ({len(segmenter.segment)})", (80, 80, 255)
        elif feature is not None:
            status, status_color = "ready - start signing", (200, 200, 200)
        else:
            status, status_color = "no hand detected", (0, 0, 255)
        cv2.putText(frame, status, (30, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if last_ranked is not None and time.time() - last_time < RESULT_HOLD:
            best_lab, best_sim = last_ranked[0]
            confident = best_sim >= 0.85          # cosine sim threshold, not a DTW margin
            color  = (0, 255, 0) if confident else (0, 160, 255)
            runner = "   ".join(f"{l} {s:.2f}" for l, s in last_ranked[1:])
            texts  = [
                (best_lab, (30, 15), 70, color),
                (f"sim {best_sim:.2f}" + ("" if confident else "  (uncertain)"),
                 (30, 100), 24, (255, 255, 255)),
                (runner, (30, 130), 20, (160, 160, 160)),
            ]
            frame = draw_cyrillic(frame, texts)

        cv2.imshow("MK Sign Language — Words (encoder)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
