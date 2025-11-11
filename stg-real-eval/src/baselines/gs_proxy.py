import numpy as np
from PIL import Image


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - dmin) / (dmax - dmin)).astype(np.float32)


def _count_peaks(sig: np.ndarray) -> int:
    if sig.size < 3:
        return 1
    diffs = np.diff(sig)
    rising = np.hstack((diffs > 0, [False]))
    falling = np.hstack(([False], diffs < 0))
    peaks = rising[:-1] & falling[1:]
    count = int(peaks.sum())
    return max(1, count)


def infer_gs_proxy(pil_img: Image.Image, depth_np: np.ndarray):
    d = _normalize_depth(depth_np)
    h, w = d.shape

    median_d = np.median(d)
    mask = (d <= median_d).astype(np.uint8)

    gy = np.abs(np.gradient(d, axis=0))
    gx = np.abs(np.gradient(d, axis=1))
    v_profile = gy.mean(axis=1)
    h_profile = gx.mean(axis=0)

    floors = _count_peaks(v_profile)
    repeats = _count_peaks(h_profile)

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        feats = np.zeros((1, 3), dtype=np.float32)
    else:
        sel = np.random.choice(ys.size, size=min(512, ys.size), replace=False)
        y_norm = ys[sel] / max(1, h - 1)
        x_norm = xs[sel] / max(1, w - 1)
        z_norm = d[ys[sel], xs[sel]]
        feats = np.stack([y_norm, x_norm, z_norm], axis=1).astype(np.float32)

    return {
        "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
        "repeats": [floors, repeats],
        "depth": 2,
        "proxy_mask": mask.tolist(),
        "cluster_feats": feats.tolist(),
    }
