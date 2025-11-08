import os
import json
import cv2
import numpy as np
from tqdm import tqdm


def run_bsp_depth(depth_dir, out_dir, thresh=0.05):
    """Simple depth-based binary-space partition baseline."""
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(depth_dir)), desc="BSP-Depth"):
        # Only operate on files that are true depth maps
        if not fname.endswith("_depth.npy"):
            continue

        depth_path = os.path.join(depth_dir, fname)
        depth = np.load(depth_path)

        # --- Robust handling of depth arrays ---
        if depth.ndim > 2:
            depth = depth.squeeze()
        depth = depth.astype(np.float64, copy=False)
        # ---------------------------------------

        gx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        edges = (np.abs(gx) + np.abs(gy)) > thresh
        n_splits = int(edges.sum() / 500)

        pred = {
            "rules": [f"Split_y_{i}" for i in range(n_splits)],
            "repeats": [1] * n_splits,
            "depth": 1,
            "persist_ids": [],
            "motion": [],
        }

        # Ensure consistent naming: scene_000_pred.json (no _depth_ in filename)
        base = fname.replace("_depth.npy", "_pred.json")
        out_path = os.path.join(out_dir, base)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred, f)
