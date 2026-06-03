"""
extractor_tiff.py
------------------
Extracts evenly-spaced raw float32 depth TIFF files from a ZED .svo file
using the Neural+ depth model.

Each TIFF pixel = exact distance in metres from the camera.
Use these files for:
    • Object distance measurement
    • Physical size calculation  (width/height in metres)
    • 3-D point cloud generation
    • Feeding into ML / CV pipelines

Output structure:
    <output_dir>/tiff/  0000_f000000_depth_raw.tiff  ...

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


def extract_tiff(
    svo_path: str,
    output_dir: str,
    num_frames: int = 500,
    min_depth: float = 0.2,
    max_depth: float = 20.0,
    confidence_threshold: int = 100,
    mask_left: int = 0,
    frame_indices: List[int] = None,
) -> List[Dict]:
    """
    Extract raw float32 metric depth TIFF files from a ZED .svo file.

    Args:
        svo_path:             Path to the .svo file.
        output_dir:           Root output directory. tiff/ subfolder created
                              automatically.
        num_frames:           Number of evenly-spaced frames to extract.
        min_depth:            Minimum depth in metres.
        max_depth:            Maximum depth in metres.
        confidence_threshold: ZED confidence 0–100. Lower = stricter. Default 100.
        mask_left:            Zero out this many pixels from the left edge.
                              Default 0 (disabled).
        frame_indices:        Pre-computed list of frame indices from main.py.
                              If None, computed from num_frames.

    Returns:
        List of dicts with keys:
            svo_filename, frame_number, frame_index, timestamp_ms, tiff_file
    """
    svo_path = str(Path(svo_path).resolve())
    tiff_dir = os.path.join(output_dir, "tiff")
    os.makedirs(tiff_dir, exist_ok=True)

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

    print(f"  [tiff]   Extracting {len(frame_indices)} frames …")

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

            # Replace NaN/Inf with 0
            valid_mask = np.isfinite(depth_arr)
            raw_save   = np.where(valid_mask, depth_arr, 0.0).astype(np.float32)

            # Apply left-edge mask
            if mask_left > 0:
                raw_save[:, :mask_left] = 0.0

            # ── Save TIFF ──────────────────────────────────────────────────
            tiff_name = f"{tag}_depth_raw.tiff"
            cv2.imwrite(os.path.join(tiff_dir, tiff_name), raw_save)

            records.append({
                "svo_filename": svo_name,
                "frame_number": saved_count,
                "frame_index":  current_frame,
                "timestamp_ms": ts_ms,
                "tiff_file":    tiff_name,
            })

            saved_count += 1

        current_frame += 1

    zed.close()
    print(f"  [tiff]   ✓ {saved_count} frames → tiff/")
    return records
