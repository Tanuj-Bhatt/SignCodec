"""
Day 5 — Receiver (the "decode" half of the semantic codec)

Listens on a UDP socket for 252-byte packets (63 float32 landmark values),
unpacks them, feeds them into the trained ASL classifier, and displays the
decoded letter live in a simple window.

This is the proof that the full loop works: camera -> 63 numbers -> 252-byte
packet -> network -> 63 numbers -> classifier -> text, with nothing but the
numbers crossing the wire.

Run this FIRST, then start sender.py in a second terminal.
Quit:
    ESC in the display window, or Ctrl+C in the terminal
"""

import json
import os
import socket
import struct
import sys

import cv2
import numpy as np
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_asl.keras")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels_asl.json")
HOST, PORT = "127.0.0.1", 5005
PACKET_FORMAT = "63f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
CONFIDENCE_THRESHOLD = 0.6  # below this, show "..." rather than a possibly-wrong guess


def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run train_classifier.py first.")
    if not os.path.exists(LABELS_PATH):
        sys.exit(f"Labels file not found at {LABELS_PATH}. Run train_classifier.py first.")
    model = keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        # keys were saved as ints via json.dump({int(i): cls, ...}) but json
        # always round-trips dict keys as strings, so cast back here.
        raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
    return model, label_map


def draw_display(decoded_label, confidence, packets_received, bytes_received):
    canvas = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(canvas, "RECEIVER", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if decoded_label is not None:
        text = f"{decoded_label}" if decoded_label != "..." else "..."
        color = (0, 255, 0) if decoded_label != "..." else (0, 165, 255)
        cv2.putText(canvas, text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4, cv2.LINE_AA)
        if decoded_label != "...":
            cv2.putText(canvas, f"confidence: {confidence*100:.1f}%", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "waiting for packets...", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Packets received: {packets_received}", (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Bytes received: {bytes_received} ({PACKET_SIZE} bytes/packet)", (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    model, label_map = load_model_and_labels()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(0.05)  # non-blocking-ish, so the display window stays responsive

    print(f"Listening on {HOST}:{PORT} for {PACKET_SIZE}-byte packets.")
    print("Waiting for sender.py to start sending...\n")

    packets_received = 0
    bytes_received = 0
    decoded_label = None
    confidence = 0.0

    while True:
        try:
            data, _addr = sock.recvfrom(1024)
            if len(data) != PACKET_SIZE:
                print(f"Warning: got {len(data)}-byte packet, expected {PACKET_SIZE}. Dropping.")
            else:
                coords = np.array(struct.unpack(PACKET_FORMAT, data), dtype="float32").reshape(1, -1)
                probs = model.predict(coords, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                decoded_label = label_map[best_idx] if confidence >= CONFIDENCE_THRESHOLD else "..."

                packets_received += 1
                bytes_received += len(data)
                print(f"[packet {packets_received}] {PACKET_SIZE} bytes -> decoded: "
                      f"{decoded_label} ({confidence*100:.1f}%)")
        except socket.timeout:
            pass

        canvas = draw_display(decoded_label, confidence, packets_received, bytes_received)
        cv2.imshow("Day 5 - Receiver", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print(f"\nTotal packets: {packets_received}, total bytes: {bytes_received}")


if __name__ == "__main__":
    main()