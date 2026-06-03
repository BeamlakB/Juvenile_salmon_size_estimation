"""
extractor_depth.py
-------------------
Extracts evenly-spaced colorized depth PNG frames from a ZED .svo file
using the Neural+ depth model.

Output structure:
    <output_dir>/depth/  0000_f000000_depth.png  ...

Returns a list of metadata dicts (one per saved frame) for the data table.
"""

import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    raise ImportError(
        "ZED SDK Python bindings not found.\n"
        "Install from: https://www.stereolabs.com/developers/release/"
    )


COLORMAPS = {
    "TURBO":   cv2.COLORMAP_TURBO,
    "JET":     cv2.COLORMAP_JET,
    "MAGMA":   cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA":  cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT":     cv2.COLORMAP_HOT,
    "BONE":    cv2.COLORMAP_BONE,
}


def extract_depth(
    svo_path: str,
    output_dir: str,
    num_frames: int = 500,
    min_depth: float = 0.2,
    max_depth: float = 20.0,
    confidence_threshold: int = 100,
    colormap: str = "TURBO",
    image_format: str = "png",
    mask_left: int = 0,
    frame_indices: List[int] = None,
) -> List[Dict]:
    """
    Extract colorized depth PNG frames from a ZED .svo file.

    Args:
        svo_path:             Path to the .svo file.
        output_dir:           Root output directory. depth/ subfolder created
                              automatically.
        num_frames:           Number of evenly-spaced frames to extract.
        min_depth:            Minimum depth in metres. Set close to nearest object.
        max_depth:            Maximum depth in metres. Set to furthest object of
                              interest. TIP: narrow this to your scene range
                              (e.g. 0.1–0.5 m) to use the full colormap.
        confidence_threshold: ZED confidence 0–100. Lower = stricter. Default 100.
        colormap:             Colormap name for visualization. Default TURBO.
        image_format:         "png" or "jpg".
        mask_left:            Zero out this many pixels from the left edge
                              (e.g. 400 to exclude a wall). Default 0.
        frame_indices:        Pre-computed list of frame indices from main.py.
                              If None, computed from num_frames.

    Returns:
        List of dicts with keys:
            svo_filename, frame_number, frame_index, timestamp_ms,
            depth_image, depth_min_m, depth_max_m, depth_mean_m, valid_pct
    """
    svo_path  = str(Path(svo_path).resolve())
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    # ── Open camera ───────────────────────────────────────────────────────────
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode     = False
    init_params.coordinate_units       = sl.UNIT.METER
    init_params.depth_mode             = sl.DEPTH_MODE.NEURAL
    init_params.depth_minimum_distance = min_depth
    init_params.depth_maximum_distance = max_depth

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"  [ERROR] Cannot open {svo_path}: {status}")
        return []

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold         = confidence_threshold
    runtime_params.texture_confidence_threshold = confidence_threshold
    runtime_params.enable_fill_mode             = False

    total_frames = zed.get_svo_number_of_frames()
    svo_name     = Path(svo_path).name
    cv_colormap  = COLORMAPS.get(colormap, cv2.COLORMAP_TURBO)

    # ── Frame indices ─────────────────────────────────────────────────────────
    if frame_indices is None:
        num_frames = min(num_frames, total_frames)
        if num_frames == 1:
            frame_indices = [0]
        else:
            step = (total_frames - 1) / (num_frames - 1)
            frame_indices = [round(i * step) for i in range(num_frames)]
    frame_set = set(frame_indices)

    depth_map     = sl.Mat()
    records       = []
    saved_count   = 0
    current_frame = 0

    print(f"  [depth]  Extracting {len(frame_indices)} frames …")

    while saved_count < len(frame_indices):
        err = zed.grab(runtime_params)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            current_frame += 1
            continue

        if current_frame in frame_set:
            tag   = f"{saved_count:04d}_f{current_frame:06d}"
            ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            # ── Retrieve depth ─────────────────────────────────────────────
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            depth_arr = depth_map.get_data()          # (H, W) float32 metres

            valid_mask  = np.isfinite(depth_arr)
            depth_clean = np.where(valid_mask, depth_arr, 0.0).astype(np.float32)
            depth_clean = np.clip(depth_clean, min_depth, max_depth)

            # ── Apply left-edge mask ───────────────────────────────────────
            if mask_left > 0:
                depth_arr[:, :mask_left]   = 0.0
                depth_clean[:, :mask_left] = 0.0
                valid_mask[:, :mask_left]  = False

            # ── Stats (computed before colorization) ───────────────────────
            vd = depth_arr[valid_mask]
            if vd.size > 0:
                d_min  = float(vd.min())
                d_max  = float(vd.max())
                d_mean = float(vd.mean())
                valid_pct = round(100.0 * vd.size / depth_arr.size, 2)
            else:
                d_min = d_max = d_mean = 0.0
                valid_pct = 0.0

            # ── Colorize ───────────────────────────────────────────────────
            depth_range = max_depth - min_depth
            depth_norm  = np.clip(
                (depth_clean - min_depth) / depth_range * 255.0, 0, 255
            ).astype(np.uint8)
            depth_norm[~valid_mask] = 0

            depth_color = cv2.applyColorMap(depth_norm, cv_colormap)
            depth_color[~valid_mask] = [30, 30, 30]

            # ── Annotate ───────────────────────────────────────────────────
            depth_color = _annotate(
                depth_color, min_depth, max_depth, d_min, d_max, d_mean, valid_pct
            )

            # ── Save ───────────────────────────────────────────────────────
            depth_name = f"{tag}_depth.{image_format}"
            cv2.imwrite(os.path.join(depth_dir, depth_name), depth_color)

            records.append({
                "svo_filename": svo_name,
                "frame_number": saved_count,
                "frame_index":  current_frame,
                "timestamp_ms": ts_ms,
                "depth_image":  depth_name,
                "depth_min_m":  round(d_min,  4),
                "depth_max_m":  round(d_max,  4),
                "depth_mean_m": round(d_mean, 4),
                "valid_pct":    valid_pct,
            })

            saved_count += 1

        current_frame += 1

    zed.close()
    print(f"  [depth]  ✓ {saved_count} frames → depth/")
    return records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotate(
    img: np.ndarray,
    min_d: float, max_d: float,
    d_min: float, d_max: float, d_mean: float,
    valid_pct: float,
) -> np.ndarray:
    lines = [
        f"Range: {min_d:.1f}m \u2013 {max_d:.1f}m",
        f"Scene min: {d_min:.2f}m  max: {d_max:.2f}m  mean: {d_mean:.2f}m",
        f"Valid pixels: {valid_pct:.1f}%",
    ]
    font, scale, th, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 6
    for i, line in enumerate(lines):
        y = 20 + i * 22
        cv2.putText(img, line, (pad + 1, y + 1), font, scale, (0,   0,   0),   th + 1)
        cv2.putText(img, line, (pad,     y),     font, scale, (255, 255, 255), th)
    return img
