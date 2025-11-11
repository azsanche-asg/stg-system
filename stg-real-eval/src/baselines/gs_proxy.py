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
    if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - dmin) / (dmax - dmin + 1e-8)).astype(np.float32)


def _load_or_compute_depth(rgb_path: Path, cache_dir: Path) -> Optional[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{rgb_path.stem}.npz"
    if npz_path.exists():
        try:
            arr = np.load(npz_path)["depth"]
            return _normalize_depth(arr)
        except Exception:
            pass
    candidates = [
        "stg_baselines.utils.midas",
        "stg_real_eval.utils.midas",
        "stg_stsg_model.utils.midas",
        "stg_synthetic_eval.utils.midas",
    ]
    for mod in candidates:
        try:
            module = __import__(mod, fromlist=["run_midas_single"])
            run_midas_single = getattr(module, "run_midas_single", None)
            if run_midas_single is None:
                continue
            depth = run_midas_single(rgb_path)
            if depth is not None:
                depth = _normalize_depth(depth)
                np.savez_compressed(npz_path, depth=depth)
                return depth
        except Exception:
            continue
    return None


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n /= (np.linalg.norm(n) + 1e-8)
    d = float(np.dot(n, centroid))
    return n.astype(np.float32), d


def _ransac_plane_from_depth(depth: np.ndarray, iters: int = 5, sample_ratio: float = 0.05,
                             inlier_thresh: float = 0.01) -> Tuple[np.ndarray, float, float]:
    H, W = depth.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = (xx / max(W - 1, 1) - 0.5) * 2.0
    yn = (yy / max(H - 1, 1) - 0.5) * 2.0
    pts = np.stack([yn, xn, depth], axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] < 100:
        return np.array([0, 0, 1], dtype=np.float32), 0.0, 0.0
    N = pts.shape[0]
    k = max(int(N * sample_ratio), 300)
    rng = np.random.default_rng(12345)
    best_score = -1.0
    best_n, best_d = np.array([0, 0, 1], dtype=np.float32), 0.0
    for _ in range(iters):
        idx = rng.choice(N, size=min(k, N), replace=False)
        n, d = _fit_plane_svd(pts[idx])
        dist = np.abs(pts @ n - d)
        score = (dist < inlier_thresh).mean()
        if score > best_score:
            best_score, best_n, best_d = score, n, d
    return best_n, float(best_d), float(best_score)


def _depth_gradients(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    gy, gx = np.gradient(d)
    return gy.astype(np.float32), gx.astype(np.float32)


def _stripe_profiles(depth: np.ndarray) -> Dict[str, np.ndarray]:
    gy, gx = _depth_gradients(depth)
    row_profile = np.mean(np.abs(gx), axis=0)
    col_profile = np.mean(np.abs(gy), axis=1)
    return {"row": row_profile, "col": col_profile}


def _count_peaks(profile: np.ndarray) -> int:
    if profile.size < 8:
        return 1
    k = max(5, profile.size // 64)
    if k % 2 == 0:
        k += 1
    sm = np.convolve(profile, np.ones(k, dtype=np.float32) / float(k), mode="same")
    sgn = np.sign(np.diff(sm))
    peaks = int(((sgn[:-1] > 0) & (sgn[1:] < 0)).sum())
    return max(1, peaks)


def _structure_masks_from_counts(H: int, W: int, rx: int, sy: int) -> Dict[str, np.ndarray]:
    rx = max(1, int(rx)); sy = max(1, int(sy))
    stripe_w = max(1, W // rx)
    band_h = max(1, H // sy)
    rep = np.zeros((H, W), np.uint8)
    for i in range(rx):
        x0 = i * stripe_w
        x1 = W if i == rx - 1 else (i + 1) * stripe_w
        rep[:, x0:x1] = 255 if (i % 2 == 0) else 0
    split = np.zeros((H, W), np.uint8)
    for j in range(sy):
        y0 = j * band_h
        y1 = H if j == sy - 1 else (j + 1) * band_h
        split[y0:y1, :] = 255 if (j % 2 == 0) else 0
    return {"Repeat_x": rep, "Split_y": split}


def _save_mask(path: Path, arr: np.ndarray):
    if _HAS_PIL:
        Image.fromarray(arr).save(path)
    else:
        np.save(path.with_suffix(".npy"), arr.astype(np.uint8))


def run_gs_proxy_for_frame(rgb_path: Path, depth_cache: Path, sample_points: int = 512) -> Dict:
    depth = _load_or_compute_depth(rgb_path, depth_cache)
    if depth is None or depth.size == 0:
        H = W = 64
        depth = np.zeros((H, W), dtype=np.float32)
        rx = sy = 1
    else:
        depth = _normalize_depth(depth)
        prof = _stripe_profiles(depth)
        rx = _count_peaks(prof["row"])
        sy = _count_peaks(prof["col"])
    H, W = depth.shape
    masks = _structure_masks_from_counts(H, W, rx, sy)
    ys, xs = np.mgrid[0:H, 0:W]
    ys = ys.flatten(); xs = xs.flatten()
    rng = np.random.default_rng(0)
    sel = rng.choice(ys.size, size=min(sample_points, ys.size), replace=False)
    feats = np.stack([
        ys[sel] / max(1, H - 1),
        xs[sel] / max(1, W - 1),
        depth[ys[sel], xs[sel]],
    ], axis=1).astype(np.float32)
    return {
        "rules": [f"Split_y_{sy}", f"Repeat_x_{rx}"],
        "repeats": [sy, rx],
        "depth": 2,
        "proxy_mask": masks["Split_y"].tolist(),
        "cluster_feats": feats.tolist(),
    }


def export_frame_results(rgb_path: Path, out_dir: Path, depth_cache: Path):
    out = run_gs_proxy_for_frame(rgb_path, depth_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = np.array(out["proxy_mask"], dtype=np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _save_mask(out_dir / f"{rgb_path.stem}_Split_y.png", mask)
    with open(out_dir / f"{rgb_path.stem}_cues.json", "w") as f:
        json.dump(out, f, indent=2)
*** End Scripts
