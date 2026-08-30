"""
Day 2, Part B, step 3 (rewritten) — Filter the full INCLUDE download down to the
INCLUDE-50 subset.

The real on-disk layout (confirmed from your actual download) is:

    include_raw/
        Colours_1of2/
            Colours/
                47. Red/
                    MVI_xxxx.MOV
                    ...
                48. Green/
                ...
        Colours_2of2/
            Colours/
                50. Yellow/
                ...
        Adjectives_1of8/
            Adjectives/
                ...
        ...

Signs are split unpredictably across the numbered chunk folders (e.g. some Colours
signs are in Colours_1of2, others in Colours_2of2) — there's no way to know which
chunk holds which sign without downloading. So instead of trying to guess a path,
this script WALKS the entire include_raw tree once, builds an index of every
"<NN>. <SignName>" folder it finds (wherever it is), and matches against that index.

Output shape (same as before):
    include50_videos/
        Yellow/
            MVI_5194.MOV
        Dog/
            MVI_3030.MOV
        ...

Run:
    python filter_include50.py --videos_dir include_raw --output include50_videos
"""

import argparse
import os
import re
import shutil
import sys

SIGN_FOLDER_PATTERN = re.compile(r"^\d+\.\s*.+$")  # e.g. "50. Yellow"
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def clean_label(raw_label):
    """'50. Yellow' -> 'Yellow'  (strip the leading numeric prefix)"""
    return re.sub(r"^\d+\.\s*", "", raw_label).strip()


def build_sign_folder_index(videos_dir):
    """Walk the whole tree once, return {raw_label_folder_name: full_path}."""
    index = {}
    for root, dirs, _files in os.walk(videos_dir):
        for d in dirs:
            if SIGN_FOLDER_PATTERN.match(d):
                index[d] = os.path.join(root, d)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True, help="include_raw root folder")
    parser.add_argument("--output", default="include50_videos")
    args = parser.parse_args()

    if not os.path.isdir(args.videos_dir):
        sys.exit(f"videos_dir not found: {args.videos_dir}")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency. Run: pip install datasets")

    print("Indexing sign folders under", args.videos_dir, "(this walks the whole tree once)...")
    sign_index = build_sign_folder_index(args.videos_dir)
    print(f"Found {len(sign_index)} sign folders on disk.\n")

    print("Loading INCLUDE metadata from Hugging Face (small metadata file, not the videos)...")
    ds = load_dataset("ai4bharat/INCLUDE")

    include50_labels = set()
    for split_name in ds.keys():
        for row in ds[split_name]:
            if row.get("include_50"):
                include50_labels.add(row["label"])  # e.g. "50. Yellow"

    print(f"Found {len(include50_labels)} unique signs belonging to INCLUDE-50.\n")
    if not include50_labels:
        sys.exit("No include_50 labels found — the metadata schema may have changed. Stop and tell Claude.")

    os.makedirs(args.output, exist_ok=True)
    copied_signs, copied_videos, missing_labels = 0, 0, []

    for raw_label in sorted(include50_labels):
        src_dir = sign_index.get(raw_label)
        if src_dir is None:
            missing_labels.append(raw_label)
            continue

        clean = clean_label(raw_label)
        dst_dir = os.path.join(args.output, clean)
        os.makedirs(dst_dir, exist_ok=True)

        video_count = 0
        for fname in os.listdir(src_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            video_count += 1

        print(f"  {raw_label} -> {clean}/  ({video_count} videos)")
        copied_signs += 1
        copied_videos += video_count

    print(f"\nCopied {copied_signs} signs, {copied_videos} total videos, into {args.output}/")
    if missing_labels:
        print(f"\n{len(missing_labels)} signs from the metadata were not found on disk:")
        for label in missing_labels:
            print("  ", label)
        print("This can happen if a few files are missing from the Zenodo mirror — normal in small numbers.")


if __name__ == "__main__":
    main()