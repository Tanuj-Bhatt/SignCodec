"""
Day 2, Part B, step 4 — Extract hand landmarks from INCLUDE-50 videos.

Videos are slower to process than photos, so instead of running MediaPipe on every
frame, this samples a handful of frames per video (around the middle, where the sign
is most likely fully formed) and keeps the first one where a hand is confidently
detected. One row per video in the output CSV — same 63-column + label schema as
process_asl_images.py, so both datasets train identically in Day 3-4.

Expects input shaped as:
    include50_videos/
        Yellow/
            MVI_5194.MOV
        Dog/
            MVI_3030.MOV
        ...
(exactly what filter_include50.py produces)

Run:
    python process_include_videos.py --input include50_videos --output datasets/include_landmarks.csv
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}
SAMPLE_FRACTIONS = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]  # try middle first, then expand outward


def build_detector():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model file not found at {MODEL_PATH}. See SETUP.md step 2.")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_video(detector, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    for frac in SAMPLE_FRACTIONS:
        frame_idx = int(total_frames * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            coords = []
            for lm in result.hand_landmarks[0]:
                coords.extend([lm.x, lm.y, lm.z])
            cap.release()
            return coords

    cap.release()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to include50_videos (one subfolder per sign)")
    parser.add_argument("--output", default="datasets/include_landmarks.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}")

    detector = build_detector()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

    total, detected, skipped = 0, 0, 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label in sorted(os.listdir(args.input)):
            label_dir = os.path.join(args.input, label)
            if not os.path.isdir(label_dir):
                continue

            label_detected = 0
            for fname in sorted(os.listdir(label_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VIDEO_EXTENSIONS:
                    continue
                total += 1
                coords = extract_landmarks_from_video(detector, os.path.join(label_dir, fname))
                if coords is None:
                    skipped += 1
                    continue
                writer.writerow(coords + [label])
                detected += 1
                label_detected += 1

            print(f"[{label}] {label_detected} videos processed successfully")

    print(f"\nFinished. total_videos={total} detected={detected} skipped={skipped}")
    print(f"Wrote {args.output}")
    if total > 0:
        print(f"Detection rate: {detected/total*100:.1f}%")


if __name__ == "__main__":
    main()