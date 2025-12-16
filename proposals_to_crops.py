#!/usr/bin/env python3
"""
proposals_to_crops.py

Usage:
  python proposals_to_crops.py --frames outputs/frames --proposals outputs/proposals/proposals.csv --out outputs/crops --max-per-frame 60 --pad 0.1 --resize 224

What it does:
 - Reads proposals CSV with columns: frame_index,filename,x1,y1,x2,y2,score,method
 - Loads frames from --frames directory
 - Optionally performs per-frame NMS to reduce overlapping boxes
 - Crops boxes with padding and optional resize
 - Saves crops to out/<frame_index>/crop_<i>__score_0.34.jpg
 - Writes crops_metadata.csv and contact sheets (per-frame and sampled global)

Notes:
 - Padding is fractional (0.1 means 10% of box width/height added around the box)
 - IoU NMS threshold default 0.5 (set to 0 to disable NMS)
"""
import os
import argparse
import csv
import math
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_proposals_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast numeric fields
            rows.append({
                "frame_index": int(r["frame_index"]),
                "filename": r["filename"],
                "x1": int(float(r["x1"])),
                "y1": int(float(r["y1"])),
                "x2": int(float(r["x2"])),
                "y2": int(float(r["y2"])),
                "score": float(r.get("score", 0.0)),
                "method": r.get("method", "")
            })
    return rows

# Simple NMS (axis-aligned boxes)
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def nms_boxes(boxes, scores, iou_threshold=0.5, max_keep=None):
    # boxes: list of [x1,y1,x2,y2]
    idxs = np.argsort(scores)[::-1]  # high to low
    keep = []
    for i in idxs:
        b = boxes[i]
        skip = False
        for k in keep:
            if iou(b, boxes[k]) > iou_threshold:
                skip = True
                break
        if not skip:
            keep.append(i)
        if max_keep and len(keep) >= max_keep:
            break
    return keep

def crop_and_save(img_path, bbox, pad, resize_to, out_path):
    """
    img_path: path to full frame
    bbox: (x1,y1,x2,y2)
    pad: fractional padding (0.1 = 10% of width/height)
    resize_to: int or None (square side)
    out_path: where to save crop (jpg)
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_w = int(round(w * pad))
    pad_h = int(round(h * pad))
    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(W, x2 + pad_w)
    y2p = min(H, y2 + pad_h)
    crop = img.crop((x1p, y1p, x2p, y2p))
    if resize_to:
        crop = crop.resize((resize_to, resize_to), Image.BILINEAR)
    crop.save(out_path, quality=92)
    return out_path, (x1p, y1p, x2p, y2p)

def create_contact_sheet(image_paths, out_path, thumb_size=(224,224), cols=6):
    if len(image_paths) == 0:
        return None
    thumbs = []
    from PIL import Image
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb_size)
            thumbs.append(im)
        except Exception as e:
            print("Warning: failed to open crop:", p, e)
    if not thumbs:
        return None
    n = len(thumbs)
    rows = math.ceil(n / cols)
    w, h = thumbs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), color=(20,20,20))
    for i, th in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet.paste(th, (c * w, r * h))
    sheet.save(out_path)
    return out_path

# -------------------------
# Main: process proposals -> crops
# -------------------------
def process(proposals_csv, frames_dir, out_dir, pad=0.1, resize_to=224, nms_iou=0.5, max_per_frame=100, overwrite=False, sample_global=36):
    ensure_dir(out_dir)
    rows = read_proposals_csv(proposals_csv)
    # group by frame_index
    frames_map = {}
    for r in rows:
        frames_map.setdefault(r['frame_index'], []).append(r)

    crops_meta = []
    all_sampled_crops = []

    # iterate frames sorted
    for frame_index in tqdm(sorted(frames_map.keys()), desc="Cropping frames"):
        proposals = frames_map[frame_index]
        # sort by score desc
        proposals = sorted(proposals, key=lambda x: x['score'], reverse=True)
        boxes = [[p['x1'], p['y1'], p['x2'], p['y2']] for p in proposals]
        scores = [p['score'] for p in proposals]

        # apply NMS to reduce overlap
        keep_idx = list(range(len(boxes)))
        if nms_iou and nms_iou > 0 and len(boxes) > 0:
            keep_idx = nms_boxes(boxes, scores, iou_threshold=nms_iou, max_keep=max_per_frame)
        else:
            keep_idx = list(range(min(len(boxes), max_per_frame)))

        # prepare frame path
        orig_fname = proposals[0]['filename']
        frame_path = os.path.join(frames_dir, orig_fname)
        if not os.path.exists(frame_path):
            print(f"Warning: frame file missing: {frame_path} (skipping frame {frame_index})")
            continue

        # output per-frame folder
        frame_out_dir = os.path.join(out_dir, f"frame_{frame_index:06d}")
        ensure_dir(frame_out_dir)
        frame_crop_paths = []

        for i_local, idx in enumerate(keep_idx):
            p = proposals[idx]
            x1,y1,x2,y2 = p['x1'], p['y1'], p['x2'], p['y2']
            score = p['score']
            crop_name = f"crop_{i_local:03d}__score_{score:.3f}__method_{p['method']}.jpg"
            out_path = os.path.join(frame_out_dir, crop_name)
            if os.path.exists(out_path) and not overwrite:
                # skip re-crop, but still add to metadata
                crops_meta.append({
                    "crop_id": os.path.relpath(out_path, out_dir),
                    "frame_index": frame_index,
                    "filename": out_path,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": score, "method": p['method'], "orig_filename": orig_fname
                })
                frame_crop_paths.append(out_path)
                continue
            try:
                saved_path, padded_box = crop_and_save(frame_path, (x1,y1,x2,y2), pad, resize_to, out_path)
                crops_meta.append({
                    "crop_id": os.path.relpath(saved_path, out_dir),
                    "frame_index": frame_index,
                    "filename": saved_path,
                    "x1": padded_box[0], "y1": padded_box[1], "x2": padded_box[2], "y2": padded_box[3],
                    "score": score, "method": p['method'], "orig_filename": orig_fname
                })
                frame_crop_paths.append(saved_path)
                all_sampled_crops.append(saved_path)
            except Exception as e:
                print(f"Error cropping frame {frame_index} box {x1,y1,x2,y2}: {e}")

        # write a per-frame contact sheet for quick visual check
        sheet_path = os.path.join(frame_out_dir, f"contact_sheet_frame_{frame_index:06d}.png")
        create_contact_sheet(frame_crop_paths, sheet_path, thumb_size=(224,224), cols=6)

    # write global sampled contact sheet
    # sample up to sample_global crops evenly
    sampled = all_sampled_crops[::max(1, int(len(all_sampled_crops) / sample_global))][:sample_global] if all_sampled_crops else []
    ensure_dir(out_dir)
    create_contact_sheet(sampled, os.path.join(out_dir, "all_crops_contact_sheet.png"), thumb_size=(160,160), cols=6)

    # write metadata CSV
    meta_csv = os.path.join(out_dir, "crops_metadata.csv")
    with open(meta_csv, "w", newline='') as f:
        fieldnames = ["crop_id", "frame_index", "filename", "x1", "y1", "x2", "y2", "score", "method", "orig_filename"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in crops_meta:
            writer.writerow(m)

    print(f"Wrote {len(crops_meta)} crops. Metadata: {meta_csv}")
    print(f"Global contact sheet: {os.path.join(out_dir,'all_crops_contact_sheet.png')}")
    return meta_csv

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True, help="Path to proposals CSV (frame_index,filename,x1,y1,x2,y2,score,method)")
    ap.add_argument("--frames", required=True, help="Directory with extracted frames")
    ap.add_argument("--out", default="outputs/crops", help="Output directory to save crops & metadata")
    ap.add_argument("--pad", type=float, default=0.1, help="Fractional padding around bbox (0.1 = 10%)")
    ap.add_argument("--resize", type=int, default=224, help="Resize crops to this size (square). 0 or None to skip resizing")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="Per-frame NMS IoU threshold (set 0 to disable)")
    ap.add_argument("--max-per-frame", type=int, default=100, help="Max crops per frame after NMS")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing crops")
    ap.add_argument("--sample-global", type=int, default=36, help="Number of crops sampled for global contact sheet")
    return ap.parse_args()

def main():
    args = parse_args()
    resize_val = int(args.resize) if (args.resize and args.resize > 0) else None
    process(args.proposals, args.frames, args.out, pad=args.pad, resize_to=resize_val,
            nms_iou=args.nms_iou, max_per_frame=args.max_per_frame, overwrite=args.overwrite, sample_global=args.sample_global)

if __name__ == "__main__":
    main()
