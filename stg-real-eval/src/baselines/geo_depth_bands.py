"""
Geometry (Depth-Bands) baseline — Block B.
Pure NumPy geometry proxy using MiDaS depth cache.
Computes quantized-depth and gradient-band masks per frame, then
temporal ΔSim and Replay IoU across frames.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(depth.min()), float(depth.max())
    if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) < 1e-6:
        return np.zeros_like(depth, np.float32)
    return ((depth - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)


def _resize_nn(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = shape
    Hs, Ws = arr.shape
    ys = (np.linspace(0, Hs - 1, Ht)).astype(int)
    xs = (np.linspace(0, Ws - 1, Wt)).astype(int)
    return arr[ys[:, None], xs[None, :]]


def _load_depth(rgb_path: Path, cache_dir: Path) -> Optional[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".npy", ".npz"):
        f = cache_dir / f"{rgb_path.stem}_midas{ext}"
        if f.exists():
            try:
                arr = np.load(f)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr["depth"]
                if arr.ndim == 3:
                    arr = arr[0]
                return _normalize_depth(arr)
            except Exception:
                continue
    return None


def _quantize_depth(depth: np.ndarray, k: int = 5) -> np.ndarray:
    qs = np.quantile(depth, np.linspace(0, 1, k + 1))
    bands = np.digitize(depth, qs[1:-1])
    return bands.astype(np.uint8)


def _gradient_mag(depth: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(depth)
    g = np.sqrt(gx ** 2 + gy ** 2)
    return _normalize_depth(g)


def _histogram(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    h, _ = np.histogram(arr, bins=bins, range=(0, 1), density=True)
    return h.astype(np.float32)


def _plane_normal(depth: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx / max(W - 1, 1) - 0.5) * 2
    yn = (yy / max(H - 1, 1) - 0.5) * 2
    pts = np.stack([xn, yn, depth], -1).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        return np.array([0, 0, 1], np.float32)
    c = pts.mean(0)
    X = pts - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-8)
    return n.astype(np.float32)


def _descriptor(depth: np.ndarray) -> np.ndarray:
    g = _gradient_mag(depth)
    d_hist = _histogram(depth, 16)
    g_hist = _histogram(g, 16)
    n = _plane_normal(depth)
    mean, std = depth.mean(), depth.std()
    return np.concatenate([[mean, std], d_hist, g_hist, n]).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        Ht, Wt = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = a[:Ht, :Wt]
        b = b[:Ht, :Wt]
    inter = np.logical_and(a == b, a > 0).sum()
    uni = np.logical_or(a > 0, b > 0).sum()
    return inter / uni if uni > 0 else np.nan


def run_sequence(frame_paths: List[Path], depth_cache: Path) -> Dict:
    target_hw = (256, 256)
    descs, bands = [], []
    for fp in frame_paths:
        d = _load_depth(fp, depth_cache)
        if d is None:
            d = np.zeros(target_hw, np.float32)
        if d.shape != target_hw:
            d = _resize_nn(d, target_hw)
        bands.append(_quantize_depth(d, 5))
        descs.append(_descriptor(d))

    frame_sims, replay_ious = [], []
    for i in range(1, len(frame_paths)):
        frame_sims.append(_cos(descs[i - 1], descs[i]))
        replay_ious.append(_iou(bands[i - 1], bands[i]))

    return {
        "frame_sims": frame_sims,
        "replay_ious": replay_ious,
        "delta_similarity": float(np.nanmean(frame_sims)) if frame_sims else np.nan,
        "replay_iou": float(np.nanmean(replay_ious)) if replay_ious else np.nan,
    }


def main_from_paths(frame_paths: List[str], depth_cache: str, out_json: str):
    fps = [Path(p) for p in frame_paths]
    out = run_sequence(fps, Path(depth_cache))
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
