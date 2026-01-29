#!/usr/bin/env python3
# Simple ROI extractor (v3) - minimal & robust
# - Mask = non-white pixels (keeps dark lines + colored heatmaps/colorbars)
# - Optional border-frame removal using flood fill from corners
# - Contours -> padded crops
#
# Usage:
#   python simple_roi.py --image input.jpg --outdir crops
#   python simple_roi.py --image sheet.jpg --outdir crops --rm-border --min-area 0.03
#   python simple_roi.py --image heatmap.png --outdir crops --min-area 0.01
#
# Dependencies: opencv-python, numpy

import os, argparse
import cv2
import numpy as np

def pad_box(x1, y1, x2, y2, pad_frac, H, W):
    w, h = x2 - x1, y2 - y1
    px, py = int(pad_frac * w), int(pad_frac * h)
    return max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)

def nonwhite_mask(img, s_thresh=12, v_thresh=245):
    # \"\"\"Return 0/255 mask of anything not near-white (colored OR dark).\"\"\"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]; v = hsv[:,:,2]
    mask = np.where((s > s_thresh) | (v < v_thresh), 255, 0).astype(np.uint8)
    # light clean
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def remove_border_connected(mask):
    # \"\"\"Kill any blobs connected to the outer page background via flood fill.\"\"\"
    H, W = mask.shape
    inv = 255 - mask  # background approx white => high
    ff = inv.copy()
    cv2.floodFill(ff, None, (0,0), 128)  # mark background from corner
    bg = (ff == 128).astype(np.uint8) * 255
    # foreground = mask but not background-connected lines
    clean = cv2.bitwise_and(mask, cv2.bitwise_not(bg))
    return clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--outdir", required=True, help="directory to save crops")
    ap.add_argument("--min-area", type=float, default=0.02, help="min ROI area as fraction of image area")
    ap.add_argument("--pad", type=float, default=0.02, help="padding around each ROI (fraction of ROI size)")
    ap.add_argument("--rm-border", action="store_true", help="remove border-connected frames/background via flood fill")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    H, W = img.shape[:2]

    mask = nonwhite_mask(img)            # keeps colors + dark lines
    if args.rm_border:
        mask = remove_border_connected(mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_px = args.min_area * H * W
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area_px:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        boxes = [(0, 0, W, H)]

    boxes.sort(key=lambda b: (b[1], b[0]))

    base = os.path.splitext(os.path.basename(args.image))[0]
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, args.pad, H, W)
        crop = img[y1:y2, x1:x2]
        outp = os.path.join(args.outdir, f"{base}_roi{i}_{x1}-{y1}-{x2}-{y2}.png")
        cv2.imwrite(outp, crop)
    print(f"Saved {len(boxes)} ROI crops to {args.outdir}")

if __name__ == "__main__":
    main()
