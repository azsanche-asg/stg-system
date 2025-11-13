import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _normalize01(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    dmin, dmax = float(depth.min()), float(depth.max())
    if not np.isfinite(dmin) or not np.isfinite(dmax) or (dmax - dmin) < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - dmin) / (dmax - dmin + 1e-8)


def _bands_from_floors(depth01: np.ndarray, floors: int) -> List[np.ndarray]:
    floors = max(1, int(floors))
    qs = np.linspace(0, 1, floors + 1)
    edges = np.quantile(depth01, qs)
    edges[0] = -1e6
    edges[-1] = 1e6
    labels = np.digitize(depth01, edges[1:-1])
    masks = [(labels == i) for i in range(floors)]
    return masks


def _repeat_columns(mask: np.ndarray, repeats: int) -> np.ndarray:
    repeats = max(1, int(repeats))
    if repeats == 1:
        return mask
    H, W = mask.shape
    cols = (np.arange(W) * repeats // max(1, W)).astype(int)
    keep = (cols % repeats) == 0
    return mask & keep[None, :]


def _feat_band(img_rgb: np.ndarray, depth01: np.ndarray, band: np.ndarray) -> np.ndarray:
    band = band.astype(bool)
    if band.sum() < 8:
        return np.zeros(10, dtype=np.float32)
    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) / 255.0
    hist, _ = np.histogram(gray[band], bins=8, range=(0, 1), density=True)
    dvals = depth01[band]
    ys, xs = np.nonzero(band)
    H, W = gray.shape
    cx = xs.mean() / max(1, W - 1)
    cy = ys.mean() / max(1, H - 1)
    feat = np.concatenate([
        hist.astype(np.float32),
        np.array([dvals.mean(), dvals.std(), cx, cy], dtype=np.float32),
    ])
    return feat


def infer_gs_guided(pil_img: Image.Image, depth_np: np.ndarray, floors: int, repeats: int) -> Dict[str, Any]:
    img = np.array(pil_img.convert("RGB"))
    d01 = _normalize01(depth_np)
    H, W = d01.shape

    masks = _bands_from_floors(d01, floors)
    guided = []
    for m in masks:
        g = _repeat_columns(m.astype(bool), repeats)
        guided.append(g)

    proxy = np.zeros((H, W), np.uint8)
    for g in guided:
        proxy |= g.astype(np.uint8)

    feats = [_feat_band(img, d01, g) for g in guided]
    if feats:
        arr = np.stack(feats, axis=0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        sims = (arr / norms) @ (arr / norms).T
        k = sims.shape[0]
        avg_sim = float(np.mean(sims[np.triu_indices(k, 1)])) if k > 1 else 0.0
    else:
        avg_sim = 0.0

    return {
        "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
        "repeats": [int(floors), int(repeats)],
        "depth": 2,
        "proxy_mask": (proxy * 255).tolist(),
        "slot_masks": [g.tolist() for g in guided],
        "cluster_feats": [f.tolist() for f in feats],
        "avg_sim": avg_sim,
    }
*** End Scripts
