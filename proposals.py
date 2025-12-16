#!/usr/bin/env python3
"""
proposals.py

Usage examples:
  python proposals.py --frames outputs/frames --out outputs/proposals --mode yolov8 --max-frames 200
  python proposals.py --frames outputs/frames --out outputs/proposals --mode selective --max-frames 100

Modes:
 - yolov8    : uses ultralytics YOLOv8n (best if you can install ultralytics)
 - selective : uses selective search fallback (pure CPU)

Outputs:
 - outputs/proposals/proposals.csv
 - optionally: outputs/proposals_samples/annotated_frame_*.jpg
"""

import os
import argparse
import csv
import json
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Try imports for optional dependencies
_HAS_ULTRALYTICS = False
_HAS_SELECTIVESEARCH = False
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

try:
    import selectivesearch
    _HAS_SELECTIVESEARCH = True
except Exception:
    _HAS_SELECTIVESEARCH = False

# ---------------------
# Helper utils
# ---------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_frames(frames_dir, max_frames=None):
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if max_frames:
        files = files[:max_frames]
    return files

# ---------------------
# YOLO proposals (ultralytics)
# ---------------------
class YoloProposer:
    def __init__(self, model_name='yolov8n.pt', device='cpu', conf_thres=0.1):
        if not _HAS_ULTRALYTICS:
            raise ImportError("ultralytics not installed. pip install ultralytics to use YOLO mode.")
        self.model = YOLO(model_name)
        self.device = device
        self.conf_thres = conf_thres

    def propose(self, img_bgr, max_boxes=50):
        """
        img_bgr: numpy array BGR (as read by cv2)
        returns list of (x1,y1,x2,y2,score)
        """
        # YOLO ultralytics accepts numpy images
        results = self.model(img_bgr, device=self.device, imgsz=640, conf=self.conf_thres, verbose=False)
        boxes = []
        if len(results) > 0:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]
                    conf = float(box.conf[0].cpu().numpy().tolist()) if hasattr(box, "conf") else 1.0
                    boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))
        # sort by score desc
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)[:max_boxes]
        return boxes

# ---------------------
# Selective Search proposals (fallback)
# ---------------------
def selective_search_propose(img_bgr, max_boxes=200, min_size=200):
    """
    img_bgr: numpy BGR image
    returns list of (x1,y1,x2,y2,score) with score set to area for ranking
    Requires 'selectivesearch' python package.
    """
    if not _HAS_SELECTIVESEARCH:
        raise ImportError("selectivesearch not installed. pip install selectivesearch")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lbl, regions = selectivesearch.selective_search(img_rgb, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    boxes = []
    H, W = img_rgb.shape[:2]
    for r in regions:
        if 'rect' not in r:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # filter out small or duplicate
        if w * h < min_size:
            continue
        if (x, y, w, h) in candidates:
            continue
        candidates.add((x, y, w, h))
        x1, y1, x2, y2 = x, y, x + w, y + h
        # clip
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        area = (x2 - x1) * (y2 - y1)
        boxes.append((x1, y1, x2, y2, float(area)))
    # sort by area (largest first) and keep top K
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)[:max_boxes]
    return boxes

# ---------------------
# Visualization utilities
# ---------------------
def draw_boxes_on_image(img_bgr, boxes, out_path, max_to_draw=50, color=(0,255,0), thickness=2):
    img = img_bgr.copy()
    H, W = img.shape[:2]
    # Draw up to max_to_draw boxes
    for i, (x1, y1, x2, y2, score) in enumerate(boxes[:max_to_draw]):
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, thickness)
        txt = f"{score:.2f}"
        cv2.putText(img, txt, (x1i, max(0,y1i-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return out_path

# ---------------------
# Main runner
# ---------------------
def run_proposals(frames_dir, out_dir, mode='yolov8', max_frames=None,
                  max_per_frame=80, min_area=100, sample_visuals=8, yolo_conf=0.1):
    ensure_dir(out_dir)
    samples_dir = os.path.join(os.path.dirname(out_dir), "proposals_samples")
    ensure_dir(samples_dir)

    frames = list_frames(frames_dir, max_frames)
    print(f"Found {len(frames)} frames in {frames_dir}")

    # instantiate proposer if needed
    yolo = None
    if mode == "yolov8":
        if not _HAS_ULTRALYTICS:
            print("ultralytics not installed; switching to selective_search fallback.")
            mode = "selective"
        else:
            print("Initializing YOLOv8n proposer (device=cpu).")
            yolo = YoloProposer(model_name='yolov8n.pt', device='cpu', conf_thres=yolo_conf)

    output_csv = os.path.join(out_dir, "proposals.csv")
    csv_file = open(output_csv, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_index", "filename", "x1", "y1", "x2", "y2", "score", "method"])

    # Visual sample indices (spread evenly)
    sample_idxs = set()
    if len(frames) > 0:
        step = max(1, len(frames) // sample_visuals)
        sample_idxs = set(list(range(0, len(frames), step))[:sample_visuals])

    for idx, fname in enumerate(tqdm(frames, desc="Proposing")):
        fpath = os.path.join(frames_dir, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print("Warning: failed to read", fpath); continue

        # choose proposer
        boxes = []
        method = mode
        try:
            if mode == "yolov8" and yolo is not None:
                boxes = yolo.propose(img_bgr, max_boxes=max_per_frame)
                method = "yolov8"
            else:
                boxes = selective_search_propose(img_bgr, max_boxes=max_per_frame, min_size=min_area)
                method = "selective"
        except Exception as e:
            print(f"Proposer error (frame {fname}): {e}; falling back to selective search")
            boxes = selective_search_propose(img_bgr, max_boxes=max_per_frame, min_size=min_area)
            method = "selective"

        # write to CSV
        for (x1, y1, x2, y2, score) in boxes:
            # minimal filtering
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            writer.writerow([idx, fname, int(x1), int(y1), int(x2), int(y2), float(score), method])

        # save annotated sample if requested
        if idx in sample_idxs:
            annotated_path = os.path.join(samples_dir, f"annotated_{idx:06d}_{fname}")
            draw_boxes_on_image(img_bgr, boxes, annotated_path, max_to_draw=40)

    csv_file.close()
    print("Wrote proposals CSV to:", output_csv)
    print("Annotated samples (visual checks) in:", samples_dir)
    return output_csv, samples_dir

# ---------------------
# CLI
# ---------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Directory containing extracted frames (JPEGs)")
    ap.add_argument("--out", default="outputs/proposals", help="Output directory for proposals CSV and samples")
    ap.add_argument("--mode", choices=["yolov8", "selective"], default="yolov8",
                    help="Proposal mode: 'yolov8' uses ultralytics YOLOv8n; 'selective' uses Selective Search fallback")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process (for quick tests)")
    ap.add_argument("--max-per-frame", type=int, default=80, help="Max proposals to keep per frame")
    ap.add_argument("--min-area", type=int, default=300, help="Filter proposals smaller than this area")
    ap.add_argument("--sample-visuals", type=int, default=8, help="How many annotated sample frames to save")
    ap.add_argument("--yolo-conf", type=float, default=0.1, help="YOLO confidence threshold (if using YOLO mode)")
    return ap.parse_args()

def main():
    args = parse_args()
    out_csv, samples_dir = run_proposals(
        frames_dir=args.frames,
        out_dir=args.out,
        mode=args.mode,
        max_frames=args.max_frames,
        max_per_frame=args.max_per_frame,
        min_area=args.min_area,
        sample_visuals=args.sample_visuals,
        yolo_conf=args.yolo_conf
    )

if __name__ == "__main__":
    main()
