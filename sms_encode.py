"""
Day 6 — SMS integration (the "works over the most constrained channel" proof)

Takes the same 63 landmark floats as sender.py, but instead of a raw binary
UDP packet, quantizes them to int16 (halves the size: 252 -> 126 bytes,
with negligible ~0.00005 error - not enough to change a classification)
and base64-encodes the result into an SMS-safe ASCII string.

Two modes:

  --mode simulate (default)
      Encodes and decodes locally, no network call. Use this to prove the
      encoding/decoding logic is correct and to get honest byte/character
      counts for your paper, without needing an SMS gateway account.

  --mode send
      Actually sends via the Fast2SMS API (https://www.fast2sms.com/).
      Requires:
        - a Fast2SMS account and API key
        - set the environment variable FAST2SMS_API_KEY before running
        - a verified recipient number (India numbers, per Fast2SMS)
      NOTE: this path has NOT been tested end-to-end here (no network
      access to fast2sms.com in this environment) - test it yourself with
      a real API key before relying on it for your demo video. If the
      gateway is flaky close to your deadline, the --mode simulate proof
      is still a fully honest way to demonstrate the concept: the encoding
      is real, only the last-mile SMS transport is being simulated.

Run:
    python sms_encode.py --mode simulate
    python sms_encode.py --mode send --phone 9XXXXXXXXX
"""

import argparse
import base64
import os
import struct
import sys

import numpy as np

SCALE = 10000  # int16 quantization scale — gives ~0.0001 precision, verified negligible error
NUM_FLOATS = 63


def encode_landmarks(coords):
    """63 floats -> base64 ASCII string, safe to put in an SMS body."""
    if len(coords) != NUM_FLOATS:
        raise ValueError(f"Expected {NUM_FLOATS} floats, got {len(coords)}")
    quantized = np.round(np.array(coords) * SCALE).astype(np.int16)
    packed = struct.pack(f"{NUM_FLOATS}h", *quantized)  # 126 bytes
    return base64.b64encode(packed).decode("ascii")


def decode_landmarks(b64_string):
    """base64 ASCII string -> 63 floats, reversing encode_landmarks."""
    packed = base64.b64decode(b64_string)
    quantized = np.array(struct.unpack(f"{NUM_FLOATS}h", packed))
    return (quantized / SCALE).tolist()


def send_via_fast2sms(message, phone):
    import requests  # imported here so --mode simulate doesn't need it installed

    api_key = os.environ.get("FAST2SMS_API_KEY")
    if not api_key:
        sys.exit("Set the FAST2SMS_API_KEY environment variable first.")

    resp = requests.post(
        "https://www.fast2sms.com/dev/bulkV2",
        headers={"authorization": api_key},
        json={"route": "q", "message": message, "language": "english", "flash": 0, "numbers": phone},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "send"], default="simulate")
    parser.add_argument("--phone", help="recipient phone number, required for --mode send")
    args = parser.parse_args()

    # demo payload — same shape as a real detected hand, values well within
    # the real observed coordinate ranges (x,y ~ 0-1.2, z ~ -0.5 to 0.5)
    demo_coords = [round(0.3 + 0.01 * i, 4) for i in range(NUM_FLOATS)]

    encoded = encode_landmarks(demo_coords)
    raw_bytes = NUM_FLOATS * 4          # original float32 packet (sender.py)
    quantized_bytes = NUM_FLOATS * 2    # after int16 quantization
    text_chars = len(encoded)           # what actually goes in the SMS body

    print("=== Encoding proof ===")
    print(f"Original packet (float32, sender.py): {raw_bytes} bytes")
    print(f"Quantized packet (int16):              {quantized_bytes} bytes")
    print(f"Base64-encoded (SMS text):             {text_chars} characters")
    print(f"Standard SMS capacity: 160 chars (GSM-7) / 70 chars (Unicode)")
    segments_needed = -(-text_chars // 160)  # ceil division
    print(f"SMS segments needed: {segments_needed} (be exact about this in the paper)")

    decoded = decode_landmarks(encoded)
    max_error = max(abs(a - b) for a, b in zip(demo_coords, decoded))
    print(f"\nRound-trip max error after quantization: {max_error:.6f} (negligible)")

    if args.mode == "simulate":
        print("\n[SIMULATE MODE] No network call made. Encoding/decoding verified locally.")
        return

    if not args.phone:
        sys.exit("--mode send requires --phone")

    print(f"\n[SEND MODE] Sending real SMS to {args.phone} via Fast2SMS...")
    result = send_via_fast2sms(encoded, args.phone)
    print("Fast2SMS response:", result)


if __name__ == "__main__":
    main()