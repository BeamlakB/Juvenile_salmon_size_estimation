"""
main.py
--------
Orchestrates extraction of RGB images, colorized depth PNGs, and raw depth
TIFFs from one or more ZED .svo files, then writes a combined CSV data table.

Each extractor runs independently but uses the SAME pre-computed frame indices
so every left image, depth PNG, and TIFF are guaranteed to match by frame.

Output structure (per SVO):
    <output>/
    └── <svo_stem>/
        ├── left/    0000_f000000_left.png    ...
        ├── right/   0000_f000000_right.png   ...
        ├── depth/   0000_f000000_depth.png   ...
        ├── tiff/    0000_f000000_depth_raw.tiff ...
        └── data.csv

A combined CSV across all SVOs is also written to <output>/all_frames.csv

Usage:
    # Single file
    python main.py --input recording.svo --output ./frames

    # Multiple files
    python main.py --input /path/to/*.svo --output ./frames

    # Select which extractors to run
    python main.py --input recording.svo --output ./frames \\
        --skip-images          # skip left/right RGB
        --skip-depth           # skip colorized depth PNG
        --skip-tiff            # skip raw TIFF

    # Full options
    python main.py --input recording.svo --output ./frames \\
        --count 300 \\
        --min-depth 0.1 --max-depth 0.5 \\
        --confidence 80 \\
        --mask-left 400 \\
        --colormap INFERNO \\
        --format png
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict

# ── Import the three extractor modules ────────────────────────────────────────
try:
    from extractor_images import extract_images, compute_frame_indices
    from extractor_depth  import extract_depth
    from extractor_tiff   import extract_tiff
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import extractor module: {e}\n"
        "Make sure extractor_images.py, extractor_depth.py, and "
        "extractor_tiff.py are in the same directory as main.py."
    )


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────

# Column order in the CSV
CSV_COLUMNS = [
    "svo_filename",
    "frame_number",
    "frame_index",
    "timestamp_ms",
    "left_image",
    "right_image",
    "depth_image",
    "tiff_file",
]


def merge_records(
    img_records:   List[Dict],
    depth_records: List[Dict],
    tiff_records:  List[Dict],
) -> List[Dict]:
    """
    Merge records from the three extractors by (svo_filename, frame_index).
    Any extractor that was skipped contributes empty strings for its columns.
    """
    # Build lookup by frame_index for depth and tiff
    depth_by_idx = {r["frame_index"]: r for r in depth_records}
    tiff_by_idx  = {r["frame_index"]: r for r in tiff_records}
    img_by_idx   = {r["frame_index"]: r for r in img_records}

    # Union of all frame indices seen
    all_indices = sorted(
        set(img_by_idx) | set(depth_by_idx) | set(tiff_by_idx)
    )

    rows = []
    for fi in all_indices:
        img   = img_by_idx.get(fi,   {})
        depth = depth_by_idx.get(fi, {})
        tiff  = tiff_by_idx.get(fi,  {})

        # Use whichever record has the base fields
        base = img or depth or tiff
        rows.append({
            "svo_filename": base.get("svo_filename", ""),
            "frame_number": base.get("frame_number", ""),
            "frame_index":  fi,
            "timestamp_ms": base.get("timestamp_ms", ""),
            "left_image":   img.get("left_image",    ""),
            "right_image":  img.get("right_image",   ""),
            "depth_image":  depth.get("depth_image", ""),
            "tiff_file":    tiff.get("tiff_file",    ""),
        })
    return rows


def write_csv(rows: List[Dict], csv_path: str) -> None:
    """Write merged rows to a CSV file."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not  os.path.isfile(csv_path):
            writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv]    ✓ {len(rows)} rows → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-SVO processing
# ──────────────────────────────────────────────────────────────────────────────

def process_svo(svo_path: str, out_root: str, args) -> List[Dict]:
    """
    Run all enabled extractors on a single SVO file.
    Returns merged rows for the all_frames.csv.
    """
    stem    = Path(svo_path).stem
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  SVO  : {Path(svo_path).name}")
    print(f"  Out  : {out_dir}")
    print(f"{sep}")

    # ── Compute shared frame indices once ─────────────────────────────────────
    print("  Computing frame indices …")
    try:
        frame_indices = compute_frame_indices(svo_path, args.count)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []
    print(f"  → {len(frame_indices)} frames selected from SVO")

    img_records   = []
    depth_records = []
    tiff_records  = []

    # ── Images ────────────────────────────────────────────────────────────────
    if not args.skip_images:
        img_records = extract_images(
            svo_path      = svo_path,
            output_dir    = out_dir,
            image_format  = args.format,
            frame_indices = frame_indices,
        )
    else:
        print("  [images] skipped")

    # ── Depth PNGs ────────────────────────────────────────────────────────────
    if not args.skip_depth:
        depth_records = extract_depth(
            svo_path             = svo_path,
            output_dir           = out_dir,
            min_depth            = args.min_depth,
            max_depth            = args.max_depth,
            confidence_threshold = args.confidence,
            colormap             = args.colormap,
            image_format         = args.format,
            mask_left            = args.mask_left,
            frame_indices        = frame_indices,
        )
    else:
        print("  [depth]  skipped")

    # ── Raw TIFFs ─────────────────────────────────────────────────────────────
    if not args.skip_tiff:
        tiff_records = extract_tiff(
            svo_path             = svo_path,
            output_dir           = out_dir,
            min_depth            = args.min_depth,
            max_depth            = args.max_depth,
            confidence_threshold = args.confidence,
            mask_left            = args.mask_left,
            frame_indices        = frame_indices,
        )
    else:
        print("  [tiff]   skipped")

    # ── Merge and write per-SVO CSV ───────────────────────────────────────────
    rows = merge_records(img_records, depth_records, tiff_records)
    write_csv(rows, os.path.join(out_dir, "data.csv"))

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract RGB images, colorized depth PNGs, and raw depth TIFFs "
            "from ZED .svo files, then write a CSV data table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    p.add_argument("--input", "-i", nargs="+", required=True,
                   help="Path(s) to .svo file(s). Supports glob patterns, e.g. videos/*.svo")
    p.add_argument("--output", "-o", default="./output",
                   help="Root output directory.")
    p.add_argument("--count", "-n", type=int, default=500,
                   help="Number of evenly-spaced frames to extract per SVO.")

    # ── Depth settings ────────────────────────────────────────────────────────
    p.add_argument("--min-depth", type=float, default=0.2, metavar="M",
                   help=(
                       "Minimum depth in metres. "
                       "Set close to your nearest object (ZED min ~0.2 m)."
                   ))
    p.add_argument("--max-depth", type=float, default=20.0, metavar="M",
                   help=(
                       "Maximum depth in metres. "
                       "TIP: narrow to your scene — e.g. 0.5 for a close-up scene — "
                       "so the full colormap range is used and TIFFs aren't mostly zero."
                   ))
    p.add_argument("--confidence", type=int, default=100, metavar="0-100",
                   choices=range(0, 101),
                   help=(
                       "ZED depth confidence threshold (0–100). "
                       "Lower = stricter (fewer, more accurate pixels)."
                   ))

    # ── Masking ───────────────────────────────────────────────────────────────
    p.add_argument("--mask-left", type=int, default=0, metavar="PX",
                   help=(
                       "Zero out this many pixels from the left edge of depth "
                       "outputs (depth PNG + TIFF). Useful to exclude a wall. "
                       "RGB images are not masked."
                   ))

    # ── Visualization ─────────────────────────────────────────────────────────
    p.add_argument("--colormap", default="TURBO",
                   choices=["TURBO","JET","MAGMA","INFERNO","PLASMA","VIRIDIS","HOT","BONE"],
                   help="Colormap for the depth PNG visualization.")
    p.add_argument("--format", default="png", choices=["png", "jpg"],
                   help="Image format for RGB and colorized depth outputs.")

    # ── Skip flags ────────────────────────────────────────────────────────────
    p.add_argument("--skip-images", action="store_true",
                   help="Skip extracting left/right RGB images.")
    p.add_argument("--skip-depth",  action="store_true",
                   help="Skip extracting colorized depth PNGs.")
    p.add_argument("--skip-tiff",   action="store_true",
                   help="Skip extracting raw depth TIFFs.")

    return p.parse_args()


def resolve_inputs(raw_inputs: List[str]) -> List[str]:
    paths = []
    for pattern in raw_inputs:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pattern):
            paths.append(pattern)
        else:
            print(f"[WARN] No file matched: {pattern}")
    return sorted(set(paths))


def main():
    args = parse_args()

    if args.min_depth >= args.max_depth:
        sys.exit(
            f"[ERROR] --min-depth ({args.min_depth}) must be less than "
            f"--max-depth ({args.max_depth})."
        )

    if args.skip_images and args.skip_depth and args.skip_tiff:
        sys.exit("[ERROR] All extractors are skipped — nothing to do.")

    svo_files = resolve_inputs(args.input)
    if not svo_files:
        sys.exit("[ERROR] No .svo files found. Check your --input paths.")

    print(f"\n{'═'*62}")
    print(f"  ZED SVO Extractor")
    print(f"{'═'*62}")
    print(f"  Files      : {len(svo_files)} SVO file(s)")
    print(f"  Output     : {args.output}")
    print(f"  Frames     : {args.count} per SVO")
    print(f"  Depth range: {args.min_depth} m – {args.max_depth} m")
    print(f"  Confidence : {args.confidence} / 100")
    print(f"  Mask left  : {args.mask_left} px" if args.mask_left else
          f"  Mask left  : off")
    print(f"  Colormap   : {args.colormap}")
    print(f"  Extractors : "
          f"{'images ' if not args.skip_images else ''}"
          f"{'depth '  if not args.skip_depth  else ''}"
          f"{'tiff'    if not args.skip_tiff   else ''}")
    print(f"{'═'*62}")

    all_rows = []

    for svo in svo_files:
        rows = process_svo(svo, args.output, args)
        all_rows.extend(rows)

    # ── Write combined CSV across all SVOs ────────────────────────────────────
    if all_rows:
        combined_path = os.path.join(args.output, "all_frames.csv")
        write_csv(all_rows, combined_path)

    print(f"\n{'═'*62}")
    print(f" All {len(svo_files)} SVO(s) processed")
    print(f"  Total frames : {len(all_rows)}")
    print(f"  Combined CSV : {os.path.join(args.output, 'all_frames.csv')}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
