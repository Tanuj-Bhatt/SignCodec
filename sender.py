"""
Day 5 — Sender (the "transmit" half of the semantic codec)

Runs the same MediaPipe HandLandmarker as extract_landmarks.py, but instead
of just printing the 63 numbers, packs them into a fixed-size 252-byte
binary packet and sends it over UDP to the receiver.

Packet format: 63 x float32, big-endian-agnostic (native), via
struct.pack('63f', *coords) -> exactly 252 bytes (63 * 4).

This is the part of the project that makes the "252 bytes instead of a
full video frame" claim real and testable, not just a number in the paper.

Run receiver.py FIRST (it needs to be listening), then this:
    python sender.py
Quit:
    ESC in the video window
"""

import os
import socket
import struct
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
HOST, PORT = "127.0.0.1", 5005
PACKET_FORMAT = "63f"  # 63 float32s = 252 bytes exactly
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)


def build_detector():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model file not found at {MODEL_PATH}. See SETUP.md step 2.")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def landmarks_to_flat_list(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def main():
    assert PACKET_SIZE == 252, f"Expected 252-byte packets, got {PACKET_SIZE}"

    detector = build_detector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    print(f"Sending to {HOST}:{PORT}, packet size = {PACKET_SIZE} bytes.")
    print("Press ESC in the video window to quit.\n")

    packets_sent = 0
    bytes_sent = 0
    last_send_time = time.time()
    SEND_INTERVAL = 0.2  # 5 packets/sec — plenty for a static/slow-changing sign, keeps demo readable

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks and (time.time() - last_send_time) >= SEND_INTERVAL:
            hand = result.hand_landmarks[0]
            coords = landmarks_to_flat_list(hand)
            packet = struct.pack(PACKET_FORMAT, *coords)
            sock.sendto(packet, (HOST, PORT))

            packets_sent += 1
            bytes_sent += len(packet)
            last_send_time = time.time()

            cv2.putText(
                frame, f"Sent packet #{packets_sent} ({len(packet)} bytes)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA
            )

        cv2.putText(
            frame, "ESC to quit | SENDER", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Total sent: {bytes_sent} bytes ({packets_sent} packets)",
            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA
        )
        cv2.imshow("Day 5 - Sender", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Honest compression comparison, printed at the end for your writeup.
    frame_h, frame_w = frame.shape[:2]
    raw_frame_bytes = frame_h * frame_w * 3  # uncompressed RGB
    # Typical JPEG compression of a webcam frame at default quality is roughly
    # 5-15% of raw size — use a conservative (larger, harder-to-beat) estimate.
    typical_jpeg_bytes = raw_frame_bytes * 0.10

    print(f"\n=== Session summary ===")
    print(f"Packets sent: {packets_sent}")
    print(f"Total bytes sent: {bytes_sent}")
    print(f"Bytes per packet: {PACKET_SIZE}")
    print(f"\nCompression comparison (per frame, honest numbers for the paper):")
    print(f"  Raw uncompressed frame ({frame_w}x{frame_h}x3): {raw_frame_bytes:,} bytes")
    print(f"  Typical JPEG-compressed frame (~10% estimate): {typical_jpeg_bytes:,.0f} bytes")
    print(f"  Our semantic packet: {PACKET_SIZE} bytes")
    print(f"  Ratio vs raw: {raw_frame_bytes / PACKET_SIZE:,.0f}x smaller")
    print(f"  Ratio vs typical JPEG: {typical_jpeg_bytes / PACKET_SIZE:,.0f}x smaller")


if __name__ == "__main__":
    main()