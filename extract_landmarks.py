"""
Day 1 — Landmark extraction (the "semantic encoder")

Opens your webcam, runs MediaPipe's HandLandmarker on every frame, draws the
21 hand landmarks on screen, and prints the 63 raw numbers (21 points x X,Y,Z)
to the terminal in real time.

This is proof that the encoder half of the system works before we touch any
dataset, classifier, or transmission logic.

Run:
    python3 extract_landmarks.py
Quit:
    press ESC with the video window focused
"""

import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
NUM_HANDS = 1  # keep at 1 for now: simpler packet, simpler classifier


def build_detector():
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            f"Model file not found at {MODEL_PATH}\n"
            "Download it first — see SETUP.md step 2."
        )
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def landmarks_to_flat_list(hand_landmarks):
    """21 landmarks x (x, y, z) -> flat list of 63 floats."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # simple connections so it reads as a hand, not just dots
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring
        (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
        (5, 9), (9, 13), (13, 17),               # palm
    ]
    for a, b in connections:
        xa, ya = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
        xb, yb = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
        cv2.line(frame, (xa, ya), (xb, yb), (0, 200, 0), 1)


def main():
    detector = build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam. Check camera permissions / index (0).")

    print("Running. Press ESC in the video window to quit.\n")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            coords = landmarks_to_flat_list(hand)
            draw_landmarks(frame, hand)

            frame_count += 1
            if frame_count % 5 == 0:  # don't flood the terminal every single frame
                print(f"[{len(coords)} numbers] first 6: {[round(c, 3) for c in coords[:6]]}")

        cv2.putText(
            frame, "ESC to quit", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow("Day 1 - Hand Landmark Extraction", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()