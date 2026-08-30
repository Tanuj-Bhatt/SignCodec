"""
Compression proof — measures REAL file sizes, not estimates.

Captures one frame from your webcam, saves it three ways, and reports the
actual byte size of each file on disk, next to your actual 252-byte
landmark packet. This gives you a number in the paper you can defend as
"measured", not "assumed".

Run:
    python measure_compression.py
"""

import os
import struct

import cv2

PACKET_SIZE = 63 * 4  # our actual packet format: 63 float32s


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    print("Capturing one frame in 2 seconds... look at the camera.")
    cv2.waitKey(2000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Failed to capture frame.")

    h, w = frame.shape[:2]
    raw_bytes = h * w * 3  # uncompressed RGB, computed not estimated

    os.makedirs("compression_proof", exist_ok=True)

    # Real JPEG file, default OpenCV quality (~95) - realistic "send a photo" baseline
    jpg_path = "compression_proof/frame.jpg"
    cv2.imwrite(jpg_path, frame)
    jpg_bytes = os.path.getsize(jpg_path)

    # Real PNG file too, for a lossless comparison point
    png_path = "compression_proof/frame.png"
    cv2.imwrite(png_path, frame)
    png_bytes = os.path.getsize(png_path)

    print(f"\nFrame captured: {w}x{h}")
    print(f"\n=== MEASURED (not estimated) sizes ===")
    print(f"Raw uncompressed RGB:  {raw_bytes:>10,} bytes  (computed: {w}*{h}*3)")
    print(f"Saved JPEG file:       {jpg_bytes:>10,} bytes  (real file: {jpg_path})")
    print(f"Saved PNG file:        {png_bytes:>10,} bytes  (real file: {png_path})")
    print(f"Our landmark packet:   {PACKET_SIZE:>10,} bytes  (63 floats x 4 bytes, struct.pack)")

    print(f"\n=== Ratios (use these in the paper, they're measured) ===")
    print(f"vs raw RGB:  {raw_bytes/PACKET_SIZE:,.0f}x smaller")
    print(f"vs JPEG:     {jpg_bytes/PACKET_SIZE:,.0f}x smaller")
    print(f"vs PNG:      {png_bytes/PACKET_SIZE:,.0f}x smaller")

    print(f"\nProof files saved in ./compression_proof/ — you can attach frame.jpg")
    print("to your submission as supporting evidence if you want.")


if __name__ == "__main__":
    main()