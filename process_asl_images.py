"""
Day 2, Part A — Convert a folder of ASL alphabet images into a landmarks CSV.

Expects a folder structure like:
    dataset_root/
        A/
            img1.jpg
            img2.jpg
        B/
            img1.jpg
        ...

(one subfolder per letter/label, images inside). This matches how most
downloaded ASL image datasets are organized. If yours looks different after
unzipping, tell me the structure and I'll adjust the script.

For each image: run MediaPipe HandLandmarker in IMAGE mode (more accurate
than live-video mode, appropriate for single static photos), extract the
63 landmark numbers, write one row per successfully-detected image to
datasets/asl_landmarks.csv in the same schema as the INCLUDE processing
script from Day 2 Part B, so both datasets can be trained on identically.

Run:
    python process_asl_images.py --input path/to/dataset_root --output datasets/asl_landmarks.csv
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def extract_landmarks(detector, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to dataset_root (one subfolder per label)")
    parser.add_argument("--output", default="datasets/asl_landmarks.csv")
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

            for fname in sorted(os.listdir(label_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                total += 1
                coords = extract_landmarks(detector, os.path.join(label_dir, fname))
                if coords is None:
                    skipped += 1
                    continue
                writer.writerow(coords + [label])
                detected += 1

            print(f"[{label}] done — running totals: detected={detected} skipped={skipped}")

    print(f"\nFinished. total_images={total} detected={detected} skipped={skipped}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()