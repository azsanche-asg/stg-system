"""
Light 3DGS Proxy v2 — Geometry temporal baseline (final).

Key fixes:
- Accept explicit frame list (exact subset from evaluator).
- Load-or-compute MiDaS small depth with project-local cache.
- Resize depth to a fixed (H, W) for comparable masks/metrics.
- NumPy-only geometry; no OpenCV.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


_TARGET_HW = (256, 256)
_CACHE_ROOT = Path("cache") / "block_b"


def _resize_nn(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    Hs, Ws = img.shape
    ys = (np.linspace(0, Hs - 1, Ht)).astype(np.int32)
    xs = (np.linspace(0, Ws - 1, Wt)).astype(np.int32)
    return img[ys[:, None], xs[None, :]]


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(np.min(depth)), float(np.max(depth))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)


def _depth_cache_path(depth_cache_dir: Path, rgb_path: Path) -> Path:
    return depth_cache_dir / f"{rgb_path.stem}_midas.npy"


def _compute_midas_small(rgb_path: Path) -> Optional[np.ndarray]:
    try:
        import torch

        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        img = Image.open(rgb_path).convert("RGB")
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32)
        return _normalize_depth(depth)
    except Exception:
        return None


def _load_or_compute_depth(rgb_path: Path, depth_cache_dir: Path) -> Optional[np.ndarray]:
    depth_cache_dir.mkdir(parents=True, exist_ok=True)
    p = _depth_cache_path(depth_cache_dir, rgb_path)
    if p.exists():
        try:
            return _normalize_depth(np.load(p))
        except Exception:
            pass
    depth = _compute_midas_small(rgb_path)
    if depth is not None:
        np.save(p, depth.astype(np.float32))
    return depth


def _plane_normal(depth: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx / max(W - 1, 1) - 0.5) * 2
    yn = (yy / max(H - 1, 1) - 0.5) * 2
    pts = np.stack([xn, yn, depth], -1).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    if pts.shape[0] < 100:
        return np.array([0, 0, 1], np.float32)
    c = pts.mean(0)
    X = pts - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-8)
    return n.astype(np.float32)


def _binary_mask(depth: np.ndarray) -> np.ndarray:
    med = float(np.median(depth))
    return (depth <= med).astype(np.uint8)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an < 1e-8 or bn < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        Ht, Wt = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a = a[:: max(1, a.shape[0] // Ht), :: max(1, a.shape[1] // Wt)][:Ht, :Wt]
        b = b[:: max(1, b.shape[0] // Ht), :: max(1, b.shape[1] // Wt)][:Ht, :Wt]
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else float("nan")


def _descriptor(depth: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.array([depth.mean(), depth.std()], np.float32),
        _plane_normal(depth),
    ]).astype(np.float32)


def run_sequence_from_paths(frame_paths: List[Path], depth_cache_dir: Path) -> Dict:
    Ht, Wt = _TARGET_HW
    depths = []
    for fp in frame_paths:
        d = _load_or_compute_depth(fp, depth_cache_dir)
        if d is None or d.size == 0:
            depths.append(np.zeros((Ht, Wt), np.float32))
        else:
            if d.shape != (Ht, Wt):
                d = _resize_nn(d, (Ht, Wt))
            depths.append(d)

    descs = [_descriptor(d) if d.any() else np.zeros(5, np.float32) for d in depths]
    masks = [_binary_mask(d) for d in depths]

    frame_sims, replay_ious = [], []
    for i in range(1, len(frame_paths)):
        frame_sims.append(_cos(descs[i - 1], descs[i]))
        replay_ious.append(_iou(masks[i - 1], masks[i]))

    return {
        "frame_sims": [float(x) for x in frame_sims],
        "replay_ious": [float(x) if np.isfinite(x) else float("nan") for x in replay_ious],
        "delta_similarity": float(np.nanmean(frame_sims)) if frame_sims else float("nan"),
        "replay_iou": float(np.nanmean(replay_ious)) if replay_ious else float("nan"),
    }


def main_from_paths(frame_paths: List[str], depth_cache_dir: str, out_json: str):
    paths = [Path(p) for p in frame_paths]
    out = run_sequence_from_paths(paths, Path(depth_cache_dir))
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

