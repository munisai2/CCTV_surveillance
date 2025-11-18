#!/usr/bin/env python3
"""
extract_frames.py

Usage:
    python extract_frames.py --video path/to/video.mp4 --out outputs/frames --mode step --step 2
    python extract_frames.py --video path/to/video.mp4 --out outputs/frames --mode fps --fps 2

Outputs:
 - <out>/frame_00000001.jpg ...
 - <out>/frames_metadata.csv  (frame_index, filename, timestamp_seconds)
 - <out>/sample_montage.png  (contact sheet of equally spaced sampled frames)
"""

import os
import cv2
import argparse
import csv
from math import floor
from PIL import Image

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def extract_by_step(video_path, out_dir, step=2, max_frames=None):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    saved = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            fname = f"frame_{frame_idx:08d}.jpg"
            out_path = os.path.join(out_dir, fname)
            # save as JPEG
            cv2.imwrite(out_path, frame)
            timestamp = frame_idx / fps
            rows.append((frame_idx, fname, timestamp))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return rows, fps, total_frames

def extract_by_fps(video_path, out_dir, target_fps=2, max_frames=None):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / target_fps)))
    return extract_by_step(video_path, out_dir, step=step, max_frames=max_frames)

def write_metadata_csv(rows, out_dir):
    csv_path = os.path.join(os.path.dirname(out_dir), "frames_metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "filename", "timestamp_seconds"])
        for r in rows:
            writer.writerow([r[0], r[1], f"{r[2]:.6f}"])
    return csv_path

def create_contact_sheet(out_dir, montage_path, max_images=36, thumb_size=(320, 180)):
    # list saved frames sorted
    files = sorted([f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not files:
        return None
    # sample up to max_images evenly
    step = max(1, int(len(files) / max_images))
    sampled = files[::step][:max_images]
    thumbs = []
    for fname in sampled:
        p = os.path.join(out_dir, fname)
        img = Image.open(p).convert("RGB")
        img.thumbnail(thumb_size)
        thumbs.append(img)
    # compute grid
    n = len(thumbs)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    w, h = thumb_size
    sheet = Image.new("RGB", (cols * w, rows * h), color=(30, 30, 30))
    for idx, th in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        sheet.paste(th, (c * w, r * h))
    sheet.save(montage_path)
    return montage_path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--out", default="outputs/frames", help="Output directory for frames (default outputs/frames)")
    ap.add_argument("--mode", choices=["step", "fps"], default="step",
                    help="Extraction mode: 'step' (every Nth frame) or 'fps' (target frames per second)")
    ap.add_argument("--step", type=int, default=2, help="If mode=step, save every Nth frame (default 2)")
    ap.add_argument("--fps", type=float, default=2.0, help="If mode=fps, approximate this many frames per second")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional: stop after saving this many frames (for quick tests)")
    return ap.parse_args()

def main():
    args = parse_args()
    video_path = args.video
    out_dir = args.out
    ensure_dir(out_dir)
    if args.mode == "step":
        rows, fps, total = extract_by_step(video_path, out_dir, step=args.step, max_frames=args.max_frames)
    else:
        rows, fps, total = extract_by_fps(video_path, out_dir, target_fps=args.fps, max_frames=args.max_frames)
    csv_path = write_metadata_csv(rows, out_dir)
    montage_path = os.path.join(os.path.dirname(out_dir), "sample_montage.png")
    create_contact_sheet(out_dir, montage_path)
    print(f"Video: {video_path}")
    print(f"Source FPS: {fps}, Estimated total frames in video: {total}")
    print(f"Saved {len(rows)} frames -> {out_dir}")
    print(f"Metadata CSV: {csv_path}")
    print(f"Sample montage (visual check): {montage_path}")

if __name__ == "__main__":
    main()
