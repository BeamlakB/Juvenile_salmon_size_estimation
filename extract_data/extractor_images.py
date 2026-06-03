"""
extractor_images.py
--------------------
Extracts evenly-spaced left and right RGB frames from a ZED .svo file.

Output structure:
    <output_dir>/left/  0000_f000000_left.png  ...
    <output_dir>/right/ 0000_f000000_right.png ...

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


def extract_images(
    svo_path: str,
    output_dir: str,
    num_frames: int = 500,
    image_format: str = "png",
    frame_indices: List[int] = None,
) -> List[Dict]:
    """
    Extract left + right RGB frames from a ZED .svo file.

    Args:
        svo_path:      Path to the .svo file.
        output_dir:    Root output directory. left/ and right/ subfolders
                       are created automatically.
        num_frames:    Number of evenly-spaced frames to extract.
        image_format:  "png" or "jpg".
        frame_indices: Optional pre-computed list of frame indices to extract.
                       If None, indices are computed from num_frames.
                       Pass this from main.py so all extractors hit the same frames.

    Returns:
        List of dicts with keys:
            svo_filename, frame_number, frame_index, timestamp_ms,
            left_image, right_image
    """
    svo_path = str(Path(svo_path).resolve())
    left_dir  = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir,  exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # ── Open camera ───────────────────────────────────────────────────────────
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units   = sl.UNIT.METER
    # Images only — use NONE depth mode for speed (no depth computation needed)
    init_params.depth_mode         = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"  [ERROR] Cannot open {svo_path}: {status}")
        return []

    total_frames = zed.get_svo_number_of_frames()
    svo_name     = Path(svo_path).name

    # ── Compute frame indices if not provided ─────────────────────────────────
    if frame_indices is None:
        num_frames = min(num_frames, total_frames)
        if num_frames == 1:
            frame_indices = [0]
        else:
            step = (total_frames - 1) / (num_frames - 1)
            frame_indices = [round(i * step) for i in range(num_frames)]
    frame_set = set(frame_indices)

    # ── Buffers ───────────────────────────────────────────────────────────────
    img_left  = sl.Mat()
    img_right = sl.Mat()

    records       = []
    saved_count   = 0
    current_frame = 0

    print(f"  [images] Extracting {len(frame_indices)} frames …")

    while saved_count < len(frame_indices):
        err = zed.grab()
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            current_frame += 1
            continue

        if current_frame in frame_set:
            tag = f"{saved_count:04d}_f{current_frame:06d}"

            # Timestamp in milliseconds
            ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            # Left
            zed.retrieve_image(img_left, sl.VIEW.LEFT)
            left_name = f"{tag}_left.{image_format}"
            _save_bgra(img_left, os.path.join(left_dir, left_name))

            # Right
            zed.retrieve_image(img_right, sl.VIEW.RIGHT)
            right_name = f"{tag}_right.{image_format}"
            _save_bgra(img_right, os.path.join(right_dir, right_name))

            records.append({
                "svo_filename": svo_name,
                "frame_number": saved_count,
                "frame_index":  current_frame,
                "timestamp_ms": ts_ms,
                "left_image":   left_name,
                "right_image":  right_name,
            })

            saved_count += 1

        current_frame += 1

    zed.close()
    print(f"  [images] ✓ {saved_count} frames → left/ and right/")
    return records


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_bgra(sl_mat: "sl.Mat", filepath: str) -> None:
    arr = sl_mat.get_data()                        # (H, W, 4) BGRA uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(filepath, bgr)


def compute_frame_indices(svo_path: str, num_frames: int) -> List[int]:
    """
    Open an SVO just long enough to read total_frames, then compute
    evenly-spaced indices.  Call this from main.py once and pass the
    result to all three extractors so they are perfectly in sync.
    """
    svo_path = str(Path(svo_path).resolve())
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode         = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Cannot open {svo_path} to read frame count: {status}")

    total = zed.get_svo_number_of_frames()
    zed.close()

    num_frames = min(num_frames, total)
    if num_frames == 1:
        return [0]
    step = (total - 1) / (num_frames - 1)
    return [round(i * step) for i in range(num_frames)]
