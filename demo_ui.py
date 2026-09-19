"""
Day 7 — Demo UI v2 (fullscreen, single-window, everything visible at once)

Same real pipeline as before (webcam -> landmarks -> real 252-byte UDP
packet -> classifier -> decoded letter) but:

  - runs TRUE fullscreen, auto-detecting your actual screen resolution,
    so nothing is cramped into a small corner window
  - all text/layout scales proportionally to screen size
  - also shows the SMS-safe encoding (int16 quantize + base64) computed
    live from the SAME packet that was just sent, using the exact
    encode_landmarks() function from sms_encode.py — not a separate
    simulated number, the real one for this exact frame

Nothing here is faked: the packet is real, the socket is real (loopback),
the classifier is your real trained model, and the SMS encoding is your
real, already-tested function.

Run:
    python demo_ui.py
Quit:
    ESC in the window
"""

import json
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow import keras

from collections import deque

from sms_encode import encode_landmarks, NUM_FLOATS as SMS_NUM_FLOATS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "model_asl.keras")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels_asl.json")

HOST, PORT = "127.0.0.1", 5006
PACKET_FORMAT = "63f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
CONFIDENCE_THRESHOLD = 0.6
SEND_INTERVAL = 0.2

WINDOW_NAME = "Semantic Sign Codec - Live Demo"


def make_process_dpi_aware():
    """Windows shrinks reported screen resolution under DPI scaling (125%/150%
    etc.) unless the process declares itself DPI-aware, causing the window to
    size itself to a smaller 'logical' resolution than the real screen. This
    fixes that at the source instead of trying to patch around it later."""
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def get_screen_size():
    """Best-effort real (physical-pixel) screen resolution; falls back to 1920x1080."""
    try:
        import tkinter as tk
        root = tk.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def build_hand_detector():
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


def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(LABELS_PATH):
        sys.exit("Trained model/labels not found. Run train_classifier.py first.")
    model = keras.models.load_model(CLASSIFIER_PATH)
    with open(LABELS_PATH) as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    return model, label_map


def landmarks_to_flat_list(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def draw_hand_overlay(frame, hand_landmarks, thickness, radius):
    h, w = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        xa, ya = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
        xb, yb = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
        cv2.line(frame, (xa, ya), (xb, yb), (0, 200, 0), thickness)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), -1)


def build_right_panel(panel_w, panel_h, decoded_label, confidence,
                       packets_sent, bytes_sent, raw_frame_bytes,
                       sms_bytes, sms_chars, sms_segments):
    # Positions are FRACTIONS of panel height, not fixed pixels scaled up —
    # fixed-pixel-times-scale blew past the bottom edge on tall/large
    # screens (verified: last 1-2 lines were getting clipped off-panel).
    # Fractional layout always fits, regardless of resolution.
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Font size scales off panel HEIGHT (so text stays large on a tall
    # split-screen layout); line POSITIONS are fixed fractions of panel
    # height (so they never depend on font size and can't drift past the
    # bottom edge, regardless of resolution). These are independent by
    # design — conflating them was what caused the earlier clipping bug.
    font_scale_unit = panel_h / 480.0
    thick_unit = max(1, int(font_scale_unit))

    def put(text, x_frac, y_frac, font_scale, color, thick_mult=1):
        x = int(x_frac * panel_w)
        y = int(y_frac * panel_h)
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * font_scale_unit, color,
                    max(1, int(thick_unit * thick_mult)), cv2.LINE_AA)

    def hline(y_frac):
        y = int(y_frac * panel_h)
        cv2.line(panel, (int(0.02 * panel_w), y), (panel_w - int(0.02 * panel_w), y),
                  (80, 80, 80), max(1, thick_unit))

    put("DECODED OUTPUT", 0.02, 0.055, 0.7, (255, 255, 255), 2)
    hline(0.075)

    if decoded_label is not None and decoded_label != "...":
        put(decoded_label, 0.04, 0.30, 4.0, (0, 255, 0), 6)
        put(f"confidence: {confidence*100:.1f}%", 0.04, 0.34, 0.6, (200, 200, 200))
    elif decoded_label == "...":
        put("...", 0.04, 0.30, 3.0, (0, 165, 255), 5)
        put("(low confidence, waiting)", 0.04, 0.34, 0.55, (150, 150, 150))
    else:
        put("show a sign...", 0.04, 0.30, 0.8, (120, 120, 120), 2)

    put("TRANSMISSION STATS", 0.02, 0.50, 0.6, (255, 255, 255))
    hline(0.515)

    put(f"Packets sent: {packets_sent}", 0.02, 0.56, 0.55, (0, 220, 220))
    put(f"Bytes sent (UDP): {bytes_sent}  ({PACKET_SIZE} bytes/packet)", 0.02, 0.60, 0.55, (0, 220, 220))
    if raw_frame_bytes:
        ratio = raw_frame_bytes / PACKET_SIZE
        put(f"vs. 1 raw camera frame: {ratio:,.0f}x smaller", 0.02, 0.64, 0.55, (0, 220, 220))

    put("WORKS OVER PLAIN SMS", 0.02, 0.72, 0.55, (255, 255, 255))
    if sms_bytes is not None:
        put("No internet needed - fits in a text message", 0.02, 0.77, 0.5, (0, 255, 120))
        put(f"({sms_bytes} bytes -> {sms_chars} chars, {sms_segments} SMS segment(s))",
            0.02, 0.80, 0.42, (140, 140, 140))
    else:
        put("(waiting for first packet...)", 0.02, 0.77, 0.5, (120, 120, 120))

    return panel


def main():
    make_process_dpi_aware()  # must happen BEFORE querying screen size, or it'll be wrong

    hand_detector = build_hand_detector()
    classifier, label_map = load_classifier()

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind((HOST, PORT))
    recv_sock.settimeout(0.01)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    screen_w, screen_h = get_screen_size()
    panel_w, panel_h = screen_w // 2, screen_h  # split screen exactly in half, full height

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass
    # Belt-and-braces fallback: even if the fullscreen property above doesn't
    # take effect on this OpenCV build, explicitly force the window to the
    # full real screen size and pin it to the top-left corner so it still
    # visually fills the screen.
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    print(f"Demo running fullscreen at {screen_w}x{screen_h}. Press ESC to quit.")

    packets_sent = 0
    bytes_sent = 0
    decoded_label = None
    confidence = 0.0
    last_send_time = 0.0
    raw_frame_bytes = None
    sms_bytes = sms_chars = sms_segments = None

    # Stability window: raw per-packet predictions flicker during the
    # transition into a sign (the moving hand often resembles a loose fist,
    # which the model reads as "0"). Instead of displaying every single
    # prediction, only update the on-screen label once the same class wins
    # a clear majority of the last few packets — this filters out the
    # transition noise without hiding genuine low-confidence uncertainty.
    STABILITY_WINDOW = 5
    STABILITY_MIN_AGREEMENT = 4  # out of STABILITY_WINDOW
    recent_predictions = deque(maxlen=STABILITY_WINDOW)

    label_scale = panel_h / 480.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_bytes is None:
            raw_frame_bytes = frame.shape[0] * frame.shape[1] * 3

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand_overlay(frame, hand, thickness=max(1, int(2 * (panel_w / 480))),
                               radius=max(2, int(4 * (panel_w / 480))))

            if (time.time() - last_send_time) >= SEND_INTERVAL:
                coords = landmarks_to_flat_list(hand)
                packet = struct.pack(PACKET_FORMAT, *coords)
                send_sock.sendto(packet, (HOST, PORT))
                packets_sent += 1
                bytes_sent += len(packet)
                last_send_time = time.time()

                # real SMS-safe encoding of this exact packet, using the tested function
                sms_encoded = encode_landmarks(coords)
                sms_bytes = SMS_NUM_FLOATS * 2
                sms_chars = len(sms_encoded)
                sms_segments = -(-sms_chars // 160)  # ceil division

        try:
            data, _addr = recv_sock.recvfrom(1024)
            if len(data) == PACKET_SIZE:
                coords = np.array(struct.unpack(PACKET_FORMAT, data), dtype="float32").reshape(1, -1)
                probs = classifier.predict(coords, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                raw_confidence = float(probs[best_idx])
                raw_label = label_map[best_idx] if raw_confidence >= CONFIDENCE_THRESHOLD else "..."

                recent_predictions.append((raw_label, raw_confidence))

                # Only move the displayed label once one class clearly
                # dominates the recent window — smooths out mid-transition
                # flicker instead of instantly showing every raw guess.
                if len(recent_predictions) == STABILITY_WINDOW:
                    labels_only = [p[0] for p in recent_predictions]
                    winner = max(set(labels_only), key=labels_only.count)
                    if labels_only.count(winner) >= STABILITY_MIN_AGREEMENT and winner != "...":
                        decoded_label = winner
                        # show the average confidence of the winning votes for a stable readout
                        winning_confs = [c for l, c in recent_predictions if l == winner]
                        confidence = sum(winning_confs) / len(winning_confs)
                    elif labels_only.count("...") >= STABILITY_MIN_AGREEMENT:
                        decoded_label = "..."
                    # else: keep showing the previous stable label until the
                    # window clearly agrees on something new
        except socket.timeout:
            pass

        left = cv2.resize(frame, (panel_w, panel_h))
        cv2.putText(left, "ENCODER (webcam + landmarks)",
                    (int(10 * label_scale), int(25 * label_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * label_scale, (255, 255, 255),
                    max(1, int(label_scale)), cv2.LINE_AA)

        right = build_right_panel(panel_w, panel_h, decoded_label, confidence,
                                   packets_sent, bytes_sent, raw_frame_bytes,
                                   sms_bytes, sms_chars, sms_segments)

        combined = np.hstack([left, right])
        cv2.putText(combined, "ESC to quit",
                    (int(10 * label_scale), combined.shape[0] - int(10 * label_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * label_scale, (150, 150, 150),
                    max(1, int(label_scale)), cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession totals: {packets_sent} packets, {bytes_sent} bytes sent.")


if __name__ == "__main__":
    main()