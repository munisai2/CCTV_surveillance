#!/usr/bin/env python3
"""
clip_ranker.py (patched)

Features:
 - Text-only, Image-only, and Text+Image fused queries
 - Augmented encoding for query images and for crop images
 - Optional crop augmentation via --augment_crops
 - Score fusion: final_score = alpha * clip_sim + beta * log(1 + det_score)
 - Simple ratio-test to filter ambiguous top candidates
 - Produces: ranked_results.csv, top_crops/, annotated_topk/

Usage examples:
  # Text-only
  python clip_ranker.py --metadata outputs/crops/crops_metadata.csv --crops outputs/crops --frames outputs/frames \
      --out outputs/clip_results_text --text_query "woman wearing red jacket" --augment_crops

  # Image-only (note quotes around path if it contains spaces)
  python clip_ranker.py --metadata outputs/crops/crops_metadata.csv --crops outputs/crops --frames outputs/frames \
      --out outputs/clip_results_image --image_query "user_inputs/red jacket.png" --augment_crops

  # Fused
  python clip_ranker.py --metadata outputs/crops/crops_metadata.csv --crops outputs/crops --frames outputs/frames \
      --out outputs/clip_results_fused --text_query "red jacket" --image_query "user_inputs/red jacket.png" \
      --augment_crops --alpha 1.0 --beta 0.02 --ratio_margin 0.03
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import cv2
import math
import sys

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_image(p):
    return Image.open(p).convert("RGB")

# -------------------------
# CLIP Encoder Class (with augmentations)
# -------------------------
class CLIPRanker:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cpu"):
        self.device = device
        print(f"[CLIP] Loading model {model_name} ({pretrained}) on {device} ...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            print("Failed to load CLIP model:", e)
            raise

    # Text encoding
    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().numpy()

    # Single image encoding
    def encode_image(self, img_pil):
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().numpy()

    # Augmented encoding for a query image (averaging multiple augmented embeddings)
    def encode_image_augmented(self, img_pil, n_aug=4):
        embs = []
        # base
        embs.append(self.encode_image(img_pil))
        w, h = img_pil.size
        for _ in range(max(1, n_aug)):
            # small random crop
            cx = np.random.randint(0, max(1, w//12))
            cy = np.random.randint(0, max(1, h//12))
            left = cx
            top = cy
            right = max(left + int(0.8*w), left+1)
            bottom = max(top + int(0.8*h), top+1)
            right = min(right, w)
            bottom = min(bottom, h)
            crop = img_pil.crop((left, top, right, bottom))
            if np.random.rand() > 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            # upsample if small
            min_side = min(crop.size)
            if min_side < 224:
                new_w = max(224, crop.size[0])
                new_h = max(224, crop.size[1])
                crop = crop.resize((new_w, new_h), Image.BILINEAR)
            embs.append(self.encode_image(crop))
        embs = np.stack(embs)
        out = embs.mean(axis=0)
        out = out / np.linalg.norm(out)
        return out

    # Augmented encoding for crops (more robust crop embeddings)
    def encode_image_multi(self, img_pil, n_aug=3):
        embs = []
        # original
        embs.append(self.encode_image(img_pil))
        w, h = img_pil.size
        for _ in range(max(1, n_aug)):
            # small random crop around center
            dx = max(1, int(w * 0.05))
            dy = max(1, int(h * 0.05))
            left = np.random.randint(0, dx + 1)
            top = np.random.randint(0, dy + 1)
            right = w - np.random.randint(0, dx + 1)
            bottom = h - np.random.randint(0, dy + 1)
            # ensure valid
            if right <= left or bottom <= top:
                crop = img_pil
            else:
                crop = img_pil.crop((left, top, right, bottom))
            if np.random.rand() > 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            # resize up if small (helps small CCTV crops)
            min_side = min(crop.size)
            if min_side < 224:
                crop = crop.resize((max(224, crop.size[0]), max(224, crop.size[1])), Image.BILINEAR)
            try:
                embs.append(self.encode_image(crop))
            except Exception:
                embs.append(self.encode_image(img_pil))
        embs = np.stack(embs)
        out = embs.mean(axis=0)
        out = out / np.linalg.norm(out)
        return out

# -------------------------
# Ranking pipeline
# -------------------------
def run_ranking(
        metadata_csv,
        crops_base_dir,
        frames_dir,
        out_dir,
        text_query=None,
        image_query_path=None,
        top_k=30,
        device="cpu",
        augment_crops=False,
        crop_aug_n=2,
        n_aug_query=4,
        alpha=1.0,
        beta=0.02,
        ratio_margin=0.03
    ):

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "top_crops"))
    ensure_dir(os.path.join(out_dir, "annotated_topk"))

    # 1. Load CLIP
    clipper = CLIPRanker(device=device)

    # 2. Build query embedding
    query_emb = None
    if text_query and image_query_path:
        print("[MODE] Fused text + image query")
        text_emb = clipper.encode_text(text_query)
        img = load_image(image_query_path)
        img_emb = clipper.encode_image_augmented(img, n_aug=n_aug_query)
        fused = text_emb + img_emb
        fused = fused / np.linalg.norm(fused)
        query_emb = fused
    elif text_query:
        print("[MODE] Text-only query")
        query_emb = clipper.encode_text(text_query)
    elif image_query_path:
        print("[MODE] Image-only query")
        img = load_image(image_query_path)
        query_emb = clipper.encode_image_augmented(img, n_aug=n_aug_query)
    else:
        raise ValueError("Either text_query or image_query_path must be provided")

    # 3. Load crops metadata
    print("[INFO] Loading crop metadata...")
    rows = []
    with open(metadata_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"[INFO] Found {len(rows)} crops")

    # 4. Encode crops (optionally augmented) and compute final scores
    enriched = []
    for r in tqdm(rows, desc="Ranking crops"):
        crop_rel = r["crop_id"]
        crop_path = os.path.join(crops_base_dir, crop_rel)
        if not os.path.exists(crop_path):
            # Some crops may be stored in nested form; try absolute filename field
            alt = r.get("filename", "")
            if alt and os.path.exists(alt):
                crop_path = alt
            else:
                # skip missing
                continue
        try:
            crop_img = load_image(crop_path)
        except Exception as e:
            # skip unreadable
            continue

        # encode crop robustly
        try:
            if augment_crops:
                crop_emb = clipper.encode_image_multi(crop_img, n_aug=crop_aug_n)
            else:
                crop_emb = clipper.encode_image(crop_img)
        except Exception:
            # fallback single encode
            crop_emb = clipper.encode_image(crop_img)

        clip_sim = float(np.dot(query_emb, crop_emb))
        # detector score: try to get numeric from metadata; if not present, use area
        try:
            det_score_raw = float(r.get("score", 0.0))
        except Exception:
            # try derive from bbox area
            try:
                x1 = int(float(r.get("x1", 0))); y1 = int(float(r.get("y1", 0)))
                x2 = int(float(r.get("x2", 0))); y2 = int(float(r.get("y2", 0)))
                det_score_raw = max(0.0, (x2 - x1) * (y2 - y1))
            except Exception:
                det_score_raw = 0.0

        # normalize det_score with log to compress scale
        det_score_norm = math.log(1.0 + det_score_raw)
        final_score = alpha * clip_sim + beta * det_score_norm

        enriched.append({
            "crop_id": crop_rel,
            "frame_index": int(r.get("frame_index", 0)),
            "filename": crop_path,
            "orig_filename": r.get("orig_filename", r.get("filename", "")),
            "x1": int(float(r.get("x1", 0))),
            "y1": int(float(r.get("y1", 0))),
            "x2": int(float(r.get("x2", 0))),
            "y2": int(float(r.get("y2", 0))),
            "method": r.get("method", ""),
            "det_score": det_score_raw,
            "clip_similarity": clip_sim,
            "final_score": final_score
        })

    if len(enriched) == 0:
        print("[WARN] No crops found or encoded. Exiting.")
        return

    # 5. Sort candidates by final_score (fusion)
    enriched = sorted(enriched, key=lambda x: x["final_score"], reverse=True)

    # 6. Apply ratio-test style filtering to remove ambiguous ones
    # We build top candidates by skipping items that are too close to the next ones (ambiguous)
    top_candidates = []
    consider_n = min(len(enriched), top_k * 6)
    for i in range(consider_n):
        if len(top_candidates) >= top_k:
            break
        curr = enriched[i]
        # find a second-best candidate to compare
        second = None
        for j in range(i+1, min(len(enriched), i+10)):
            if enriched[j]["orig_filename"] != curr["orig_filename"]:
                second = enriched[j]
                break
        # if no good second found, accept
        if second is None:
            top_candidates.append(curr)
            continue
        # if margin is large enough, accept
        if (curr["final_score"] - second["final_score"]) >= ratio_margin:
            top_candidates.append(curr)
        else:
            # ambiguous; skip it (helps reduce false positives)
            continue

    # fallback if not enough passed ratio test
    if len(top_candidates) < top_k:
        # fill remaining with top enriched results
        seen = set([c["crop_id"] for c in top_candidates])
        for e in enriched:
            if e["crop_id"] in seen:
                continue
            top_candidates.append(e)
            if len(top_candidates) >= top_k:
                break

    top = top_candidates[:top_k]

    # 7. Save top crops
    for i, r in enumerate(top):
        src = r["filename"]
        dst = os.path.join(out_dir, "top_crops", f"rank_{i:03d}__sim_{r['clip_similarity']:.4f}__final_{r['final_score']:.4f}.jpg")
        try:
            Image.open(src).save(dst)
        except Exception:
            # try copy raw
            try:
                import shutil
                shutil.copy(src, dst)
            except Exception:
                pass

    # 8. Annotate original frames for top matches
    for i, r in enumerate(top):
        frame_fname = r["orig_filename"]
        if not frame_fname:
            continue
        frame_path = os.path.join(frames_dir, frame_fname)
        if not os.path.exists(frame_path):
            # try if r["filename"] folder has original frame path
            continue
        try:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            txt = f"{i} | clip:{r['clip_similarity']:.3f} final:{r['final_score']:.3f}"
            cv2.putText(img, txt, (max(0, x1), max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out_path = os.path.join(out_dir, "annotated_topk", f"rank_{i:03d}.jpg")
            cv2.imwrite(out_path, img)
        except Exception:
            continue

    # 9. Save CSV (all ranked by final score)
    out_csv = os.path.join(out_dir, "ranked_results.csv")
    fieldnames = list(enriched[0].keys())
    try:
        with open(out_csv, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in enriched:
                writer.writerow(r)
        print("[DONE] Ranked CSV saved to:", out_csv)
    except Exception as e:
        print("Failed to write CSV:", e)

    print("[DONE] Top crops saved to:", os.path.join(out_dir, "top_crops"))
    print("[DONE] Annotated frames saved to:", os.path.join(out_dir, "annotated_topk"))
    return

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to crops_metadata.csv")
    ap.add_argument("--crops", required=True, help="Base directory of crops (the base used in crop_id)")
    ap.add_argument("--frames", required=True, help="Directory of original frames (for annotation)")
    ap.add_argument("--out", default="outputs/clip_results", help="Output directory")
    ap.add_argument("--text_query", type=str, default=None, help="Text query (e.g. 'woman wearing red jacket')")
    ap.add_argument("--image_query", type=str, default=None, help="Path to image query (one-shot). Quote if contains spaces.")
    ap.add_argument("--top_k", type=int, default=30, help="How many top results to save")
    ap.add_argument("--device", type=str, default="cpu", help="torch device (cpu or cuda)")
    ap.add_argument("--augment_crops", action="store_true", help="Use augmentation when encoding crops (slower but robust)")
    ap.add_argument("--crop_aug_n", type=int, default=2, help="Number of crop augmentations for crop encoding")
    ap.add_argument("--n_aug_query", type=int, default=4, help="Number of augmentations for the query image encoding")
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for CLIP similarity in final score")
    ap.add_argument("--beta", type=float, default=0.02, help="Weight for detector score (log-scaled) in final score")
    ap.add_argument("--ratio_margin", type=float, default=0.03, help="Minimal final_score gap for ratio-test filtering")
    return ap.parse_args()

def main():
    args = parse_args()
    # small path guards
    if not os.path.exists(args.metadata):
        print("Metadata file not found:", args.metadata); sys.exit(1)
    if not os.path.exists(args.crops):
        print("Crops base path not found:", args.crops); sys.exit(1)
    if not os.path.exists(args.frames):
        print("Frames path not found:", args.frames); sys.exit(1)
    if args.image_query and not os.path.exists(args.image_query):
        print("Image query file not found:", args.image_query); sys.exit(1)
    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "top_crops"))
    ensure_dir(os.path.join(args.out, "annotated_topk"))

    run_ranking(
        metadata_csv=args.metadata,
        crops_base_dir=args.crops,
        frames_dir=args.frames,
        out_dir=args.out,
        text_query=args.text_query,
        image_query_path=args.image_query,
        top_k=args.top_k,
        device=args.device,
        augment_crops=args.augment_crops,
        crop_aug_n=args.crop_aug_n,
        n_aug_query=args.n_aug_query,
        alpha=args.alpha,
        beta=args.beta,
        ratio_margin=args.ratio_margin
    )

if __name__ == "__main__":
    main()
