"""
SignCodec — Streamlit Cloud Web Application (ASL Semantic Codec)

Ultra-low bandwidth sign language communication dashboard.
Features:
- Live Real-Time WebRTC Webcam Stream with 21 3D hand landmarks
- Snapshot camera input / image upload
- Single split-screen visual dashboard (mirroring demo_ui.py)
"""

from collections import deque
import json
import os
import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from sms_encode import encode_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
ASL_MODEL_PATH = os.path.join(BASE_DIR, "model_asl.keras")
ASL_LABELS_PATH = os.path.join(BASE_DIR, "labels_asl.json")

PACKET_SIZE = 252
CONFIDENCE_THRESHOLD = 0.60
STABILITY_WINDOW = 5
STABILITY_MIN_AGREEMENT = 4

st.set_page_config(
    page_title="SignCodec - Live ASL Semantic Codec",
    page_icon="🤟",
    layout="wide",
)

# --- Cached Model & Detector Loading ---
@st.cache_resource
def load_detector():
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

@st.cache_resource
def load_classifier():
    model = keras.models.load_model(ASL_MODEL_PATH)
    with open(ASL_LABELS_PATH, "r") as f:
        raw = json.load(f)
        labels = {int(k): v for k, v in raw.items()}
    return model, labels

detector = load_detector()
classifier, label_map = load_classifier()

# Landmark connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

def draw_hand_overlay(frame, hand_landmarks, panel_w):
    h, w = frame.shape[:2]
    thickness = max(1, int(2 * (panel_w / 480.0)))
    radius = max(2, int(4 * (panel_w / 480.0)))
    for a, b in CONNECTIONS:
        xa, ya = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
        xb, yb = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
        cv2.line(frame, (xa, ya), (xb, yb), (0, 220, 100), thickness, cv2.LINE_AA)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), radius, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(1, radius - 2), (255, 255, 255), -1, cv2.LINE_AA)

def build_right_panel(panel_w, panel_h, decoded_label, confidence,
                      packets_sent, bytes_sent, raw_frame_bytes,
                      sms_bytes, sms_chars, sms_segments):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
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

    put("DECODED OUTPUT (ASL)", 0.02, 0.055, 0.7, (255, 255, 255), 2)
    hline(0.075)

    if decoded_label is not None and decoded_label != "...":
        put(decoded_label, 0.04, 0.30, 4.0, (0, 255, 0), 6)
        put(f"confidence: {confidence*100:.1f}%", 0.04, 0.35, 0.65, (200, 200, 200))
    elif decoded_label == "...":
        put("...", 0.04, 0.30, 3.0, (0, 165, 255), 5)
        put("(low confidence, stabilizing)", 0.04, 0.35, 0.55, (150, 150, 150))
    else:
        put("show an ASL sign...", 0.04, 0.30, 0.8, (120, 120, 120), 2)

    put("TRANSMISSION STATS", 0.02, 0.50, 0.6, (255, 255, 255))
    hline(0.515)

    put(f"Packets sent: {packets_sent}", 0.02, 0.56, 0.55, (0, 220, 220))
    put(f"Bytes sent (UDP): {bytes_sent}  ({PACKET_SIZE} bytes/packet)", 0.02, 0.60, 0.55, (0, 220, 220))
    if raw_frame_bytes:
        ratio = raw_frame_bytes / PACKET_SIZE
        put(f"vs. 1 raw camera frame: {ratio:,.0f}x smaller", 0.02, 0.64, 0.55, (0, 255, 120))

    put("WORKS OVER PLAIN SMS", 0.02, 0.72, 0.55, (255, 255, 255))
    hline(0.735)
    if sms_bytes is not None:
        put("No internet needed - fits in a text message", 0.02, 0.78, 0.5, (0, 255, 120))
        put(f"({sms_bytes} bytes -> {sms_chars} chars, {sms_segments} SMS segment(s))",
            0.02, 0.82, 0.45, (180, 180, 180))
    else:
        put("(waiting for detected hand...)", 0.02, 0.78, 0.5, (120, 120, 120))

    return panel

def process_frame(frame, state):
    panel_h = 480
    aspect = frame.shape[1] / frame.shape[0]
    panel_w = int(panel_h * aspect)

    left_frame = cv2.resize(frame, (panel_w, panel_h))
    raw_frame_bytes = frame.shape[0] * frame.shape[1] * 3

    # Run MediaPipe
    rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    sms_bytes = sms_chars = sms_segments = None

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_hand_overlay(left_frame, hand, panel_w)

        coords = []
        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])

        state["packets_sent"] += 1
        state["bytes_sent"] += PACKET_SIZE

        sms_encoded = encode_landmarks(coords)
        sms_bytes = 126
        sms_chars = len(sms_encoded)
        sms_segments = -(-sms_chars // 160)

        coords_arr = np.array(coords, dtype="float32").reshape(1, -1)
        probs = classifier.predict(coords_arr, verbose=0)[0]
        best_idx = int(np.argmax(probs))
        raw_conf = float(probs[best_idx])
        raw_label = label_map.get(best_idx, "...") if raw_conf >= CONFIDENCE_THRESHOLD else "..."

        state["recent_predictions"].append((raw_label, raw_conf))

        if len(state["recent_predictions"]) == STABILITY_WINDOW:
            labels_only = [p[0] for p in state["recent_predictions"]]
            winner = max(set(labels_only), key=labels_only.count)
            if labels_only.count(winner) >= STABILITY_MIN_AGREEMENT and winner != "...":
                state["decoded_label"] = winner
                winning_confs = [c for l, c in state["recent_predictions"] if l == winner]
                state["confidence"] = sum(winning_confs) / len(winning_confs)
            elif labels_only.count("...") >= STABILITY_MIN_AGREEMENT:
                state["decoded_label"] = "..."
    else:
        state["recent_predictions"].clear()
        state["decoded_label"] = None
        state["confidence"] = 0.0

    label_scale = panel_h / 480.0
    cv2.putText(left_frame, "ENCODER (webcam + landmarks)",
                (int(10 * label_scale), int(25 * label_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * label_scale, (255, 255, 255),
                max(1, int(label_scale)), cv2.LINE_AA)

    right_panel = build_right_panel(
        panel_w, panel_h,
        state["decoded_label"],
        state["confidence"],
        state["packets_sent"],
        state["bytes_sent"],
        raw_frame_bytes,
        sms_bytes, sms_chars, sms_segments,
    )

    combined = np.hstack([left_frame, right_panel])
    return combined


# --- Streamlit UI Layout ---
st.title("🤟 SignCodec: Live ASL Semantic Codec")
st.caption("Transmitting Sign Language over 252-Byte UDP Packets and 126-Byte Offline SMS | 98.8% ASL Accuracy")

tab1, tab2 = st.tabs(["📹 Live WebRTC Stream", "📸 Snapshot Camera / Upload"])

with tab1:
    st.write("Click **START** below to stream your webcam directly through WebRTC:")
    try:
        import av
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

        class WebRTCProcessor(VideoProcessorBase):
            def __init__(self):
                self.state = {
                    "packets_sent": 0,
                    "bytes_sent": 0,
                    "decoded_label": None,
                    "confidence": 0.0,
                    "recent_predictions": deque(maxlen=STABILITY_WINDOW),
                }

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                combined = process_frame(img, self.state)
                return av.VideoFrame.from_ndarray(combined, format="bgr24")

        webrtc_streamer(
            key="signcodec-stream",
            video_processor_factory=WebRTCProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.warning(f"WebRTC module notice: {e}. You can also use the Snapshot Camera in Tab 2.")

with tab2:
    st.write("Take a snapshot of an ASL sign or upload a photo:")
    col_cam, col_upload = st.columns(2)
    with col_cam:
        photo = st.camera_input("Take a photo of a hand sign")
    with col_upload:
        uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

    img_source = photo or uploaded_file
    if img_source is not None:
        file_bytes = np.asarray(bytearray(img_source.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        static_state = {
            "packets_sent": 1,
            "bytes_sent": PACKET_SIZE,
            "decoded_label": None,
            "confidence": 0.0,
            "recent_predictions": deque(maxlen=STABILITY_WINDOW),
        }
        for _ in range(STABILITY_WINDOW):
            result_canvas = process_frame(img_bgr, static_state)

        result_rgb = cv2.cvtColor(result_canvas, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="SignCodec Semantic Split-Screen Output", use_container_width=True)

st.markdown("---")
st.markdown("*GitHub Repository: [github.com/Tanuj-Bhatt/SignCodec](https://github.com/Tanuj-Bhatt/SignCodec)*")
