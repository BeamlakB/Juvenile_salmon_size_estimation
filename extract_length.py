import argparse
import csv
import os
import sys
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for batch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.graph import pixel_graph
from skimage.morphology import skeletonize
from ultralytics import SAM, YOLO

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA INTRINSICS  (edit these to match your camera)
# ══════════════════════════════════════════════════════════════════════════════
fx, fy =773.335, 773.59
cx_cam, cy_cam = 653.055, 354.457
k1, k2, k3, k4 = -0.0320652, 0.00802689, 0.0182561,-0.0147054

K = np.array([[fx,  0, cx_cam],
              [ 0, fy, cy_cam],
              [ 0,  0,      1]], dtype=np.float64)
D = np.array([k1, k2, k3, k4], dtype=np.float64)

ORIG_W, ORIG_H = 1280, 720
SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_depth_map(depth_path: str) -> np.ndarray:
    """Load a ZED depth map (.npy, .tiff, .png, …) → float32 array in mm."""
    p = str(depth_path)
    if p.endswith(".npy"):
        depth = np.load(p).astype(np.float32)
    else:
        depth = cv2.imread(p, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth map: {p}")
    return depth


def undistort_point(px: float, py: float):
    """Undistort a pixel coordinate using the fisheye camera model."""
    point = np.array([[[float(px), float(py)]]], dtype=np.float32)
    undist = cv2.fisheye.undistortPoints(point, K, D, P=K)
    return undist[0][0]


def pixel_to_3d(px: float, py: float, depth_map: np.ndarray):
    """Back-project a pixel to 3-D (mm).  Returns None if depth invalid."""
    Z = depth_map[int(py), int(px)]
    if Z <= 0 or np.isnan(Z) or np.isinf(Z):
        return None
    x_u, y_u = undistort_point(px, py)
    X = (x_u - cx_cam) * Z / fx
    Y = (y_u - cy_cam) * Z / fy
    return np.array([X, Y, Z])


def trace_skeleton_path(skeleton: np.ndarray):
    """Trace skeleton pixels from one endpoint to the other (ordered list)."""
    def neighbors(skel, y, x):
        h, w = skel.shape
        return [(y + dy, x + dx)
                for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
                and 0 <= y + dy < h and 0 <= x + dx < w
                and skel[y + dy, x + dx]]

    points = np.argwhere(skeleton)
    if len(points) < 2:
        return []

    endpoints = [(y, x) for y, x in points if len(neighbors(skeleton, y, x)) == 1]
    if len(endpoints) < 2:
        endpoints = [tuple(points[0]), tuple(points[-1])]

    start = endpoints[0]
    visited, path, current = {start}, [start], start
    while True:
        nbs = [n for n in neighbors(skeleton, *current) if n not in visited]
        if not nbs:
            break
        current = nbs[0]
        visited.add(current)
        path.append(current)
    return path


def arc_length_3d(path, depth_map: np.ndarray):
    """Sum 3-D Euclidean distances along skeleton path."""
    total_mm, valid = 0.0, 0
    for i in range(len(path) - 1):
        y0, x0 = path[i]
        y1, x1 = path[i + 1]
        p0 = pixel_to_3d(x0, y0, depth_map)
        p1 = pixel_to_3d(x1, y1, depth_map)
        if p0 is not None and p1 is not None:
            total_mm += np.linalg.norm(p1 - p0)
            valid += 1
    return total_mm, valid

def arc_length_3d_smoothed(path, depth_map, window_size=17,
                           min_z=150,        # tighter low end (ZED min range)
                            max_z=1000,       # wider high end — don't cut real fish
                            max_mm_per_px=0.9):
    """
    3D arc length with depth validity gate and mm/px plausibility check.
    """
    
    total_mm = 0.0
    coords_3d = []

    for y, x in path:
        Z = depth_map[int(y), int(x)]
        # gate 1: depth must be in valid sensor range
        if Z <= min_z or Z >= max_z or np.isnan(Z) or np.isinf(Z):
            continue
        # gate 2: mm/px at this depth must be plausible for a salmon
        mm_per_px_here = Z / fx   # ≈ pixel pitch at depth Z
        if mm_per_px_here > max_mm_per_px:
            continue              # depth value is background, not fish
        p = pixel_to_3d(x, y, depth_map)
        if p is not None:
            coords_3d.append(p)

    if len(path) < 200:
        window_size -= 2
    if len(coords_3d) < window_size:
        return 0.0, 0

    smoothed_path = []
    for i in range(len(coords_3d) - window_size + 1):
        window = coords_3d[i: i + window_size]
        smoothed_path.append(np.mean(window, axis=0))

    total_mm = sum(
        np.linalg.norm(smoothed_path[i+1] - smoothed_path[i])
        for i in range(len(smoothed_path) - 1)
    )
    return total_mm, len(smoothed_path)


def analyze_skeleton(skeleton):
    """Detect branching in skeleton — a Y-fork means two fish crossed"""
    
    # Find all skeleton pixels
    skel_pixels = np.argwhere(skeleton)
    
    # Count neighbors for each skeleton pixel
    branch_points = []
    endpoints = []
    
    for (r, c) in skel_pixels:
        # 8-connected neighbors
        neighbors = skeleton[r-1:r+2, c-1:c+2].sum() - 1  # subtract self
        if neighbors >= 3:
            branch_points.append((r, c))  # Y-junction or cross
        elif neighbors == 1:
            endpoints.append((r, c))
    
    return {
        "n_branch_points": len(branch_points),
        "n_endpoints": len(endpoints),
        "branch_locations": branch_points,
        # Single fish: 0 branch points, 2 endpoints
        # Two crossed fish: 1+ branch points, 3-4 endpoints
    }


def is_crossed_skeleton(skel_metrics):
    """A clean single fish has 0 branch points and exactly 2 endpoints"""
    return skel_metrics["n_branch_points"] > 0 or skel_metrics["n_endpoints"] > 2

# ──────────────────────────────────────────────────────────────────────────────
# Mask quality / overlap helpers
# ──────────────────────────────────────────────────────────────────────────────

def analyze_mask_shape(binary_mask: np.ndarray) -> dict:
    """Shape metrics that reveal if a mask contains 2 overlapping fish."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"solidity": 1.0, "circularity": 1.0, "aspect_ratio": 1.0,
                "max_defect_depth": 0.0, "significant_defects": 0}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    aspect_ratio = 1.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        aspect_ratio = MA / ma if ma > 0 else 1

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_idx)
    max_depth, sig_def = 0.0, 0
    if defects is not None:
        for defect in defects[:, 0]:
            d = defect[3] / 256.0
            max_depth = max(max_depth, d)
            if d > 15:
                sig_def += 1

    return {"solidity": solidity, "circularity": circularity,
            "aspect_ratio": aspect_ratio, "max_defect_depth": max_depth,
            "significant_defects": sig_def}


def is_crossed_mask(metrics: dict,
                    solidity_thresh: float = 0.80,
                    defect_thresh: float = 20.0,
                    min_defects: int = 1) -> bool:
    return (metrics["solidity"] < solidity_thresh and
            (metrics["max_defect_depth"] > defect_thresh or
             metrics["significant_defects"] >= min_defects))


def detect_overlapping_masks(mask1, mask2, iou_threshold: float = 0.5):
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)
    intersection = np.logical_and(m1, m2).sum()
    union        = np.logical_or(m1, m2).sum()
    iou          = intersection / union if union > 0 else 0
    smaller      = min(m1.sum(), m2.sum())
    overlap_ratio = intersection / smaller if smaller > 0 else 0
    is_overlap   = iou > iou_threshold or overlap_ratio > 0.5
    return is_overlap, {"iou": iou, "overlap_ratio": overlap_ratio}


def filter_masks(masks_orig: np.ndarray):
    """Remove overlapping / crossed masks; return clean mask indices."""
    n = len(masks_orig)
    dropped = set()

    for i, j in combinations(range(n), 2):
        m1 = (masks_orig[i] > 0).astype(np.uint8)
        m2 = (masks_orig[j] > 0).astype(np.uint8)
        is_overlap, _ = detect_overlapping_masks(m1, m2)
        if is_overlap:
            drop = i if masks_orig[i].sum() > masks_orig[j].sum() else j
            dropped.add(drop)

    clean = [idx for idx in range(n) if idx not in dropped]
    return clean

def keep_largest_blobs(binary_mask, min_area=None):
    """
    Removes small islands from a binary mask. 
    If min_area is None, it keeps only the single largest object.
    """
    # Ensure mask is 8-bit single channel
    binary_mask = binary_mask.astype(np.uint8)
    
    # 1. Find all connected components
    # num_labels: number of blobs found (including background)
    # labels: a map where each pixel has the ID of the blob it belongs to
    # stats: statistics for each blob (left, top, width, height, area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # If there's only 1 label, it's just the background (empty mask)
    if num_labels <= 1:
        return binary_mask

    # 2. Extract areas (the 4th column of the stats matrix)
    # We skip index 0 because that is the background area
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    # 3. Determine our threshold
    if min_area is None:
        # Keep ONLY the single largest blob
        threshold = np.max(areas)
    else:
        # Keep blobs larger than the user-defined area
        threshold = min_area

    # 4. Create the output mask
    # We create a boolean mask where the area of the label is >= threshold
    # Then map it back to the label IDs
    new_mask = np.zeros_like(binary_mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= threshold:
            new_mask[labels == i] = 255
            
    return new_mask

# ──────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ──────────────────────────────────────────────────────────────────────────────

def process_image(image_path: Path,
                  depth_path: Path,
                  yolo_model: YOLO,
                  sam_model: SAM,
                  vis_dir: Path) -> list[dict]:
    """
    Run the full pipeline on one image.
    Returns a list of result dicts (one per valid fish).
    Saves a visualisation PNG to vis_dir.
    """
    stem = image_path.stem
    print(f"\n{'='*60}")
    print(f"  Image : {image_path.name}")
    print(f"  Depth : {depth_path.name}")

    # ── Load ──────────────────────────────────────────────────
    image_orig = np.array(Image.open(image_path).convert("RGB"))
    depth_orig = load_depth_map(depth_path) * 1000   # → mm  (if stored in m)
    # If your depth is already in mm, remove * 1000

    h, w = image_orig.shape[:2]

    # ── YOLO ──────────────────────────────────────────────────
    yolo_results = yolo_model(str(image_path), conf=0.6, imgsz=1280)
    boxes_orig   = yolo_results[0].boxes.xyxy.cpu().numpy()
    print(f"  YOLO detected {len(boxes_orig)} fish")

    if len(boxes_orig) == 0:
        print("  → No detections, skipping.")
        return []

    # Filter boxes touching image border
    margin = 2
    filtered_boxes = [b for b in boxes_orig
                      if b[0] > margin and b[1] > margin
                      and b[2] < w - margin and b[3] < h - margin]

    if not filtered_boxes:
        print("  → All boxes on border, skipping.")
        return []

    # ── SAM ───────────────────────────────────────────────────
    sam_results  = sam_model(str(image_path), bboxes=filtered_boxes)
    masks_orig   = sam_results[0].masks.data.cpu().numpy()  # (N, H, W)
    print(f"  SAM produced {len(masks_orig)} masks")

    # ── Filter overlapping masks ──────────────────────────────
    n = len(masks_orig)
    dropped = set()
    for i, j in combinations(range(n), 2):
        m1 = (masks_orig[i] > 0).astype(np.uint8)
        m2 = (masks_orig[j] > 0).astype(np.uint8)
        is_ov, _ = detect_overlapping_masks(m1, m2)
        if is_ov:
            drop = i if masks_orig[i].sum() > masks_orig[j].sum() else j
            dropped.add(drop)

    clean_indices = [idx for idx in range(n) if idx not in dropped]
    print(f"  Masks after overlap filter: {len(clean_indices)} "
          f"(dropped {len(dropped)})")

    # ── Per-fish measurement ──────────────────────────────────
    fish_results = []

    for local_id, idx in enumerate(clean_indices):
        mask  = masks_orig[idx]
        box   = filtered_boxes[idx]
        binary = (mask > 0.5).astype(np.uint8)

        # Shape quality check
        shape_metrics = analyze_mask_shape(binary)
        if is_crossed_mask(shape_metrics):
            print(f"    Fish {local_id+1}: crossed mask detected — skipping")
            continue

        # Clean mask
        blurred = cv2.GaussianBlur(binary.astype(np.float32) * 255, (7, 7), 2)
        binary  = (blurred > 127).astype(np.uint8)
        binary  = ndimage.binary_fill_holes(binary).astype(np.uint8)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # smaller

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

        binary= keep_largest_blobs (binary, 500 )

        # Skeletonize
        skeleton = skeletonize(binary)
        skel_metrics = analyze_skeleton (skeleton)
        if is_crossed_skeleton(skel_metrics):
            print(f"    Fish {local_id+1}: branched skeleton detected — skipping")
            continue
        # Pixel length
        graph, _  = pixel_graph(skeleton, connectivity=2)
        length_px = graph.sum() / 2
        if length_px < 100:  # too short to be a fish
            print(f"    Fish {local_id+1}: branched skeleton detected — skipping")
            continue
        # 3-D arc length
        path              = trace_skeleton_path(skeleton)
        length_mm, valid  = arc_length_3d_smoothed(path, depth_orig)
        # After arc_length_3d_smoothed call:
        coverage = valid / max(len(path) - 1, 1)
        if coverage < 0.60:   # less than 60% of skeleton had valid depth
            print(f"    Fish {local_id+1}: low depth coverage ({coverage:.0%}) — skipping")
            continue
        print(f"    Fish {local_id+1}: {length_px:.0f} px  →  "
              f"{length_mm:.1f} mm  ({length_mm/10:.2f} cm)  "
              f"[{valid}/{max(len(path)-1,1)} valid depth steps]")

        fish_results.append({
            "image":      image_path.name,
            "fish_id":    local_id + 1,
            "length_px":  round(length_px, 1),
            "length_mm":  round(length_mm, 1),
            "length_cm":  round(length_mm / 10, 2),
            "valid_depth_steps": valid,
            # internal (not written to CSV)
            "_binary":    binary,
            "_skeleton":  skeleton,
            "_path":      path,
            "_box":       box,
            "_depth_crop": depth_orig[int(box[1]):int(box[3]),
                                      int(box[0]):int(box[2])],
            "_img_crop":  image_orig[int(box[1]):int(box[3]),
                                     int(box[0]):int(box[2])],
        })

    # ── Visualisation ─────────────────────────────────────────
    if fish_results:
        _save_visualization(fish_results, stem, vis_dir)

    return fish_results


def _save_visualization(fish_results: list, stem: str, vis_dir: Path):
    """Save a 4-column grid (original, mask, skeleton, depth) for all fish."""
    n = len(fish_results)
    fig, axes = plt.subplots(n, 4, figsize=(18, 5 * n),
                             squeeze=False)

    col_titles = ["Original + BBox", "SAM Mask", "Skeleton", "Depth (mm)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight="bold")

    for row, r in enumerate(fish_results):
        box        = r["_box"]
        x1, y1, x2, y2 = map(int, box)
        img_crop   = r["_img_crop"]
        mask_crop  = r["_binary"][y1:y2, x1:x2]
        skel_crop  = r["_skeleton"][y1:y2, x1:x2]
        depth_crop = r["_depth_crop"]

        # Col 0 — original crop
        axes[row][0].imshow(img_crop)
        axes[row][0].set_ylabel(f"Fish {r['fish_id']}", fontsize=11)

        # Col 1 — mask overlay
        axes[row][1].imshow(img_crop)
        overlay = np.zeros((*mask_crop.shape, 4))
        overlay[mask_crop > 0] = [0.2, 0.8, 0.2, 0.5]
        axes[row][1].imshow(overlay)

        # Col 2 — skeleton + path
        axes[row][2].imshow(skel_crop, cmap="gray")
        path = r["_path"]
        if path:
            pa = np.array(path)
            axes[row][2].plot(pa[:, 1] - x1, pa[:, 0] - y1,
                              "r-", linewidth=1.5)
            axes[row][2].plot(pa[0, 1]  - x1, pa[0, 0]  - y1,
                              "go", markersize=8, label="Head")
            axes[row][2].plot(pa[-1, 1] - x1, pa[-1, 0] - y1,
                              "bo", markersize=8, label="Tail")
            axes[row][2].legend(fontsize=8)
        axes[row][2].set_xlabel(
            f"{r['length_px']:.0f} px  →  "
            f"{r['length_mm']:.1f} mm  ({r['length_cm']:.2f} cm)",
            fontsize=10, color="red"
        )

        # Col 3 — depth
        dm = axes[row][3].imshow(depth_crop, cmap="plasma")
        plt.colorbar(dm, ax=axes[row][3], label="mm")

        for ax in axes[row]:
            ax.axis("off")
        axes[row][2].axis("on")
        axes[row][2].set_xticks([])
        axes[row][2].set_yticks([])

    plt.suptitle(f"Fish Measurement — {stem}",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = vis_dir / f"{stem}_vis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DEPTH PATH RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

def resolve_depth_path(image_path: Path,
                       depth_dir: Path,
                       depth_suffix: str) -> Path | None:
    """
    Build the depth map path from the image stem.

    Strategy (tries in order):
      1. Replace image stem suffix with depth_suffix directly
         e.g. 0048_f014754_left_seed512.png → 0048_f014754_depth_raw.tiff
         (takes the first two '_'-separated tokens as the common prefix)
      2. Same stem, different extension
         e.g. 0048_f014754_left_seed512.png → 0048_f014754_left_seed512.tiff
    """
    stem = image_path.stem  # e.g. "0048_f014754_left_seed512"

    # Strategy 1 — common prefix + custom suffix
    parts  = stem.split("_")
    prefix = "_".join(parts[:2]) if len(parts) >= 2 else stem
    candidate1 = depth_dir / f"{prefix}{depth_suffix}"
    if candidate1.exists():
        return candidate1

    # Strategy 2 — same stem, try all common depth extensions
    for ext in [".tiff", ".tif", ".npy", ".png"]:
        candidate2 = depth_dir / f"{stem}{ext}"
        if candidate2.exists():
            return candidate2

    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch fish length measurement pipeline")
    parser.add_argument("--images",       required=True,
                        help="Directory containing input images")
    parser.add_argument("--depths",       required=True,
                        help="Directory containing depth maps")
    parser.add_argument("--output",       required=True,
                        help="Output directory (CSV + visualizations)")
    parser.add_argument("--yolo",         required=True,
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--sam",          default="sam2_l.pt",
                        help="SAM model name or path (default: sam2_l.pt)")
    parser.add_argument("--depth-suffix", default="_depth_raw.tiff",
                        help="Suffix appended to image prefix to find depth map "
                             "(default: _depth_raw.tiff)")
    parser.add_argument("--csv",          default="fish_lengths.csv",
                        help="CSV filename inside --output (default: fish_lengths.csv)")
    args = parser.parse_args()

    image_dir  = Path(args.images)
    depth_dir  = Path(args.depths)
    output_dir = Path(args.output)
    vis_dir    = output_dir / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    print ("start stuff")
    # Collect images
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_IMG_EXTS
    ])
    print(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("No images found — check --images path and file extensions.")
        sys.exit(1)

    # Load models once
    print("\nLoading models…")
    yolo_model = YOLO(args.yolo)
    sam_model  = SAM(args.sam)

    # CSV setup
    csv_path = output_dir / args.csv
    csv_fields = ["image", "fish_id", "length_px",
                  "length_mm", "length_cm", "valid_depth_steps"]

    all_rows = []
    skipped  = []

    for img_path in image_paths:
        depth_path = resolve_depth_path(img_path, depth_dir, args.depth_suffix)

        if depth_path is None:
            print(f"\n  No depth map found for {img_path.name} — skipping.")
            skipped.append(img_path.name)
            continue

        try:
            fish_results = process_image(
                img_path, depth_path, yolo_model, sam_model, vis_dir)

            for r in fish_results:
                all_rows.append({k: r[k] for k in csv_fields})

        except Exception as exc:
            print(f"\n  Error processing {img_path.name}: {exc}")
            skipped.append(img_path.name)
            continue

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Images processed : {len(image_paths) - len(skipped)}")
    print(f"  Images skipped   : {len(skipped)}")
    print(f"  Fish measured    : {len(all_rows)}")
    print(f"  CSV saved        → {csv_path}")
    print(f"  Visualizations   → {vis_dir}")

    if skipped:
        print(f"\nSkipped files:")
        for s in skipped:
            print(f"  • {s}")

    if all_rows:
        lengths = [r["length_cm"] for r in all_rows]
        print(f"\nLength stats (cm):")
        print(f"  Min  : {min(lengths):.2f}")
        print(f"  Max  : {max(lengths):.2f}")
        print(f"  Mean : {sum(lengths)/len(lengths):.2f}")


if __name__ == "__main__":
    main()
    