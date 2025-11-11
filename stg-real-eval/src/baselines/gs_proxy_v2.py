"""
Light 3DGS Proxy v2 — Geometry temporal baseline.

This version explicitly computes frame-to-frame depth stability (Replay IoU)
and structural similarity (ΔSim) from MiDaS depth maps.
No OpenCV, only NumPy.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)


def _load_depth(rgb_path: Path) -> Optional[np.ndarray]:
    cache_root = Path("cache")
    candidates = list(cache_root.rglob(f"{rgb_path.stem}_midas.npy"))
    if not candidates:
        return None
    depth = np.load(candidates[0])
    if depth.ndim == 3:
        depth = depth[0]
    return _normalize_depth(depth)


def _plane_normal(depth: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx / max(W - 1, 1) - 0.5) * 2
    yn = (yy / max(H - 1, 1) - 0.5) * 2
    pts = np.stack([xn, yn, depth], axis=-1).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 100:
        return np.array([0, 0, 1], np.float32)
    ctr = pts.mean(0)
    X = pts - ctr
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n /= (np.linalg.norm(n) + 1e-8)
    return n.astype(np.float32)


def _binary_mask(depth: np.ndarray) -> np.ndarray:
    median = np.median(depth)
    return (depth <= median).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni > 0 else np.nan


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _descriptor(depth: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.array([depth.mean(), depth.std()], np.float32),
        _plane_normal(depth),
    ]).astype(np.float32)


def run_sequence(frames: list[Path]) -> Dict:
    descs, masks = [], []
    for fp in frames:
        d = _load_depth(fp)
        if d is None:
            descs.append(np.zeros(5, np.float32))
            masks.append(np.zeros((64, 64), np.uint8))
            continue
        descs.append(_descriptor(d))
        masks.append(_binary_mask(d))

    frame_sims, replay_ious = [], []
    for i in range(1, len(frames)):
        frame_sims.append(_cos(descs[i - 1], descs[i]))
        replay_ious.append(_iou(masks[i - 1], masks[i]))

    return {
        "frame_sims": frame_sims,
        "replay_ious": replay_ious,
        "delta_similarity": float(np.nanmean(frame_sims)) if frame_sims else np.nan,
        "replay_iou": float(np.nanmean(replay_ious)) if replay_ious else np.nan,
    }


def main(frames_dir: Path, out_path: Path):
    frames = sorted(
        [p for p in Path(frames_dir).iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )
    out = run_sequence(frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, required=True)
    args = ap.parse_args()
    main(args.frames_dir, args.out_path)
