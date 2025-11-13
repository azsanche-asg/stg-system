import numpy as np
from pathlib import Path
from typing import Any, Dict

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    mn, mx = float(arr.min()), float(arr.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-6:
        return np.zeros_like(arr, np.float32)
    return (arr - mn) / (mx - mn)


def _descriptor(img_rgb: np.ndarray, depth01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() < 16:
        return np.zeros(10, dtype=np.float32)
    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) / 255.0
    hist, _ = np.histogram(gray[mask], bins=8, range=(0, 1), density=True)
    dvals = depth01[mask]
    ys, xs = np.nonzero(mask)
    H, W = gray.shape
    cx = xs.mean() / max(1, W - 1)
    cy = ys.mean() / max(1, H - 1)
    feat = np.concatenate([
        hist.astype(np.float32),
        np.array([dvals.mean(), dvals.std(), cx, cy], np.float32),
    ])
    return (feat - feat.mean()) / (feat.std() + 1e-6)


def infer_stsg_gs_simple(pil_img: Image.Image, depth_np: np.ndarray,
                         floors: int, repeats: int) -> Dict[str, Any]:
    img = np.array(pil_img.convert("RGB"))
    depth01 = _normalize01(depth_np)
    mask = (depth01 <= np.median(depth01)).astype(np.uint8)
    if mask.sum() == 0:
        mask[:] = 1
    feat = _descriptor(img, depth01, mask)
    return {
        "rules": [f"Split_y_{max(1,int(floors))}", f"Repeat_x_{max(1,int(repeats))}"],
        "repeats": [int(max(1, floors)), int(max(1, repeats))],
        "depth": 2,
        "proxy_mask": (mask * 255).tolist(),
        "slot_masks": [mask.astype(bool).tolist()],
        "cluster_feats": [feat.tolist()],
        "avg_sim": 0.0,
    }
*** End Scripts
