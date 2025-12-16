#!/usr/bin/env python3
"""
track_from_ranked.py

Takes a ranked_results.csv from clip_ranker and starts a tracker from a chosen rank.
Then tracks forward and backward across frames and saves the track CSV and annotated frames.

Usage examples:

# Basic (top-1)
python src/track_from_ranked.py \
  --ranked outputs/clip_results/ranked_results.csv \
  --frames outputs/frames \
  --out outputs/tracks \
  --video_path path/to/video.mp4         # optional: to create annotated video

# Use a specific rank (0-indexed)
python src/track_from_ranked.py --ranked outputs/clip_results/ranked_results.csv --frames outputs/frames --out outputs/tracks --rank 2

Notes:
 - The script expects ranked CSV to contain columns:
   crop_id, frame_index, filename, orig_filename, x1, y1, x2, y2, clip_similarity, final_score (or similar)
 - If orig_filename contains full frame file name (e.g., frame_000012.jpg) the script will load frames from --frames/<orig_filename>.
 - The script uses OpenCV trackers (CSRT by default). CSRT is more accurate; switch to KCF or MOSSE for speed.
"""

import os
import csv
import argparse
from collections import OrderedDict
import cv2
from tqdm import tqdm
import numpy as np

# ------------------
# Helpers
# ------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_ranked_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_frame(frames_dir, fname):
    p = os.path.join(frames_dir, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Frame file not found: {p}")
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return img

def bbox_to_int(b):
    return int(round(float(b[0]))), int(round(float(b[1]))), int(round(float(b[2]))), int(round(float(b[3])))

def clamp_bbox(b, W, H):
    x1,y1,x2,y2 = b
    x1 = max(0, min(W-1, int(round(x1))))
    y1 = max(0, min(H-1, int(round(y1))))
    x2 = max(0, min(W-1, int(round(x2))))
    y2 = max(0, min(H-1, int(round(y2))))
    return x1,y1,x2,y2

def bbox_xywh_from_xyxy(x1,y1,x2,y2):
    return int(x1), int(y1), int(x2-x1), int(y2-y1)

# ------------------
# Tracker runner
# ------------------
def run_track_from_ranked(ranked_csv, frames_dir, out_dir, rank_index=0, tracker_type='CSRT',
                          extend_frames=200, step=1, video_path=None):
    """
    ranked_csv: path to ranked_results.csv
    frames_dir: directory containing frames (frame_xxx.jpg)
    out_dir: directory to save outputs
    rank_index: which ranked detection to start from (0-based)
    extend_frames: how many frames forward/backwards to attempt (or will stop at frame boundaries)
    step: frame step when tracking (1 = every frame)
    """

    ensure_dir(out_dir)
    rows = read_ranked_csv(ranked_csv)
    if len(rows) == 0:
        raise RuntimeError("No rows in ranked CSV.")

    if rank_index < 0 or rank_index >= len(rows):
        raise ValueError("rank_index out of range")

    chosen = rows[rank_index]
    # Expect the CSV contains frame_index, orig_filename, x1,y1,x2,y2
    frame_index = int(float(chosen.get('frame_index', chosen.get('frame_idx', 0))))
    orig_fname = chosen.get('orig_filename', chosen.get('filename', None))
    x1 = float(chosen.get('x1', chosen.get('left', 0)))
    y1 = float(chosen.get('y1', chosen.get('top', 0)))
    x2 = float(chosen.get('x2', chosen.get('right', 0)))
    y2 = float(chosen.get('y2', chosen.get('bottom', 0)))
    clip_sim = float(chosen.get('clip_similarity', chosen.get('clip_sim', 0)))
    final_score = float(chosen.get('final_score', chosen.get('score', 0)))

    if orig_fname is None:
        raise RuntimeError("ranked CSV must contain 'orig_filename' or 'filename' column to locate the frame file.")

    print(f"Starting track from rank {rank_index}: frame_index={frame_index}, file={orig_fname}, bbox=({x1},{y1},{x2},{y2})")

    # Load the seed frame
    seed_img = load_frame(frames_dir, orig_fname)
    H, W = seed_img.shape[:2]
    # clamp bbox
    x1i,y1i,x2i,y2i = clamp_bbox((x1,y1,x2,y2), W, H)
    if x2i <= x1i or y2i <= y1i:
        raise RuntimeError("Invalid bbox dimensions after clamp.")

    # Convert to tracker format (x,y,w,h)
    init_x, init_y, init_w, init_h = bbox_xywh_from_xyxy(x1i, y1i, x2i, y2i)
    print("Initial bbox (x,y,w,h):", (init_x, init_y, init_w, init_h))

    # Create tracker
    def create_tracker_by_name(name):
        """
        Automatically finds the correct constructor in:
        - cv2.TrackerX_create
        - cv2.legacy.TrackerX_create
        - very old cv2.Tracker_create('X')
        """
        # 1. Try cv2.TrackerX_create
        ctor = getattr(cv2, f"Tracker{name}_create", None)
        if callable(ctor):
            try:
                return ctor()
            except:
                pass

        # 2. Try cv2.legacy.TrackerX_create
        if hasattr(cv2, "legacy"):
            ctor = getattr(cv2.legacy, f"Tracker{name}_create", None)
            if callable(ctor):
                try:
                    return ctor()
                except:
                    pass

        # 3. Old API fallback
        if hasattr(cv2, "Tracker_create"):
            try:
                return cv2.Tracker_create(name)
            except:
                pass

        raise RuntimeError(
            f"Could not create tracker '{name}'. "
            "Check available constructors."
        )
    tracker = create_tracker_by_name(tracker_type)




    # Initialize tracker on seed frame
    tracker.init(seed_img, (init_x, init_y, init_w, init_h))

    # Tracking forward
    results = OrderedDict()  # frame_index -> dict
    # include the seed
    results[frame_index] = {
        'frame_index': frame_index,
        'filename': orig_fname,
        'x1': x1i, 'y1': y1i, 'x2': x2i, 'y2': y2i,
        'clip_similarity': clip_sim, 'final_score': final_score
    }

    # Forward tracking
    current_frame = frame_index
    # Determine maximum frame index by scanning frames_dir file names
    all_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    # Map filenames to sorted list index -> we assume frame files are of form frame_000000.jpg etc.
    filename_to_index = {fname: idx for idx, fname in enumerate(all_files)}
    # Make list for index -> filename mapping
    idx_to_fname = all_files

    max_idx = len(idx_to_fname) - 1
    # find the start position in list
    if orig_fname not in filename_to_index:
        # Try direct integer-frame indexing if files named differently
        start_pos = None
        for i, fn in enumerate(idx_to_fname):
            if fn == orig_fname:
                start_pos = i
                break
        if start_pos is None:
            raise RuntimeError("Could not find starting frame filename in frames directory")
    else:
        start_pos = filename_to_index[orig_fname]

    # Forward
    pos = start_pos
    pbar = tqdm(total=max_idx - pos, desc="Forward tracking")
    failure_count = 0
    while pos < max_idx:
        pos += step
        next_fname = idx_to_fname[pos]
        frame = cv2.imread(os.path.join(frames_dir, next_fname))
        if frame is None:
            break
        ok, bbox = tracker.update(frame)
        if not ok:
            # tracking failed on this frame: increment failure_count, but allow a few failures
            failure_count += 1
            if failure_count > 10:
                # stop forward tracking
                break
            else:
                # store NaN bbox
                results[int(pos)] = {'frame_index': int(pos), 'filename': next_fname, 'x1': None, 'y1': None, 'x2': None, 'y2': None, 'clip_similarity': None, 'final_score': None}
                pbar.update(1)
                continue
        else:
            failure_count = 0
            x, y, w, h = bbox
            x1n, y1n, x2n, y2n = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
            results[int(pos)] = {'frame_index': int(pos), 'filename': next_fname, 'x1': x1n, 'y1': y1n, 'x2': x2n, 'y2': y2n, 'clip_similarity': None, 'final_score': None}
        pbar.update(1)
    pbar.close()

    # Backward tracking: re-init tracker on seed frame, then step backwards
    tracker = create_tracker_by_name(tracker_type)

    tracker.init(seed_img, (init_x, init_y, init_w, init_h))
    pos = start_pos
    pbar2 = tqdm(total=pos, desc="Backward tracking")
    failure_count = 0
    while pos > 0:
        pos -= step
        prev_fname = idx_to_fname[pos]
        frame = cv2.imread(os.path.join(frames_dir, prev_fname))
        if frame is None:
            break
        ok, bbox = tracker.update(frame)
        if not ok:
            failure_count += 1
            if failure_count > 10:
                break
            else:
                results[int(pos)] = {'frame_index': int(pos), 'filename': prev_fname, 'x1': None, 'y1': None, 'x2': None, 'y2': None, 'clip_similarity': None, 'final_score': None}
                pbar2.update(1)
                continue
        else:
            failure_count = 0
            x, y, w, h = bbox
            x1n, y1n, x2n, y2n = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
            results[int(pos)] = {'frame_index': int(pos), 'filename': prev_fname, 'x1': x1n, 'y1': y1n, 'x2': x2n, 'y2': y2n, 'clip_similarity': None, 'final_score': None}
        pbar2.update(1)
    pbar2.close()

    # Save results to CSV and annotated frames
    csv_out = os.path.join(out_dir, f"track_rank{rank_index}.csv")
    with open(csv_out, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "filename", "x1", "y1", "x2", "y2", "clip_similarity", "final_score"])
        for k in sorted(results.keys()):
            r = results[k]
            w.writerow([r['frame_index'], r['filename'], r['x1'], r['y1'], r['x2'], r['y2'], r.get('clip_similarity'), r.get('final_score')])

    print("Wrote track CSV to:", csv_out)

    # Save annotated frames for frames that have bbox
    ann_dir = os.path.join(out_dir, "annotated_frames")
    ensure_dir(ann_dir)
    for k in sorted(results.keys()):
        r = results[k]
        if r['x1'] is None:
            continue
        img_path = os.path.join(frames_dir, r['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        cv2.rectangle(img, (r['x1'], r['y1']), (r['x2'], r['y2']), (0,0,255), 3)
        txt = f"{r['frame_index']}"
        cv2.putText(img, txt, (max(5, r['x1']), max(15, r['y1']-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        out_p = os.path.join(ann_dir, f"annot_{r['frame_index']:06d}.jpg")
        cv2.imwrite(out_p, img)

    # Optional: assemble annotated frames into a short mp4 (requires ffmpeg installed and frames named in order)
    try:
        video_out = os.path.join(out_dir, f"track_rank{rank_index}.mp4")
        # Use cv2 VideoWriter
        # find a sample annotated frame to get size
        sample_frames = sorted([f for f in os.listdir(ann_dir) if f.endswith(".jpg")])
        if len(sample_frames) > 0:
            sample = cv2.imread(os.path.join(ann_dir, sample_frames[0]))
            h, w = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(video_out, fourcc, 10.0, (w,h))
            for fname in sorted(sample_frames):
                frame = cv2.imread(os.path.join(ann_dir, fname))
                vw.write(frame)
            vw.release()
            print("Wrote annotated track video to:", video_out)
    except Exception as e:
        print("Could not write track video (error):", e)

    return csv_out, ann_dir

# ------------------
# CLI
# ------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", required=True, help="Path to ranked_results.csv")
    ap.add_argument("--frames", required=True, help="Directory with original frames (as used earlier)")
    ap.add_argument("--out", default="outputs/tracks", help="Output dir to save track CSV and annotated frames")
    ap.add_argument("--rank", type=int, default=0, help="Which rank from ranked_results.csv to start tracking from (0-index)")
    ap.add_argument("--tracker", type=str, default="CSRT", choices=["CSRT","KCF","MOSSE","MIL"], help="OpenCV tracker type")
    ap.add_argument("--step", type=int, default=1, help="Frame step when tracking (1 = every frame)")
    ap.add_argument("--video_path", type=str, default=None, help="Optional: original source video (to create output mp4). If not provided, uses frames to assemble video.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    csv_out, ann_dir = run_track_from_ranked(args.ranked, args.frames, args.out, rank_index=args.rank,
                                             tracker_type=args.tracker, step=args.step, video_path=args.video_path)
    print("Tracking done. CSV:", csv_out, "annotated frames in:", ann_dir)
