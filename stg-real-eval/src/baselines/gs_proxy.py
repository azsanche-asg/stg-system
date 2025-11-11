from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    m, M = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - m) / (M - m)


def _read_depth_file(p: Path) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p)
    # PNG/EXR fallback (PIL reads PNG; EXR support may vary)
    return np.array(Image.open(p).convert("F"))


def _load_depth_from_cache(image_path: Path, depth_cache_dir: Path) -> np.ndarray:
    """
    ONLY reads depth; never runs MiDaS.
    Tries: <stem>_midas.npy, <stem>.npy, <stem>.png, <stem>.exr in depth_cache_dir.
    If not found, try the canonical '.../midas/' path under cache/.
    Falls back to 1 - grayscale(image) so the eval won’t crash.
    """
    stem = image_path.stem
    cands = [
        depth_cache_dir / f"{stem}_midas.npy",
        depth_cache_dir / f"{stem}.npy",
        depth_cache_dir / f"{stem}.png",
        depth_cache_dir / f"{stem}.exr",
    ]

    # also look under common midas cache roots
    for root in [Path("cache") / "block_b", Path("cache")]:
        cands.extend([
            root / "**" / "midas" / f"{stem}_midas.npy",
            root / "**" / "midas" / f"{stem}.npy",
            root / "**" / "midas" / f"{stem}.png",
            root / "**" / "midas" / f"{stem}.exr",
        ])

    # try direct hits first
    for p in cands:
        if "**" in str(p):
            # best-effort recursive discovery
            try:
                for hit in p.parents[2].rglob(p.name):
                    return _normalize01(_read_depth_file(hit))
            except Exception:
                pass
        else:
            if p.exists():
                try:
                    return _normalize01(_read_depth_file(p))
                except Exception:
                    pass

    # last resort — keep pipeline alive
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    return 1.0 - gray


def _depth_bands(depth: np.ndarray, k: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
    h, w = depth.shape
    X = depth.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)  # near→far
    remap = np.zeros_like(labels)
    for rank, idx in enumerate(order):
        remap[labels == idx] = rank
    masks = [(remap.reshape(h, w) == r) for r in range(k)]
    return masks, centers[order]


def _largest_band_mask(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        return None
    areas = [int(m.sum()) for m in masks]
    return masks[int(np.argmax(areas))]


def _dominant_repeats(masks: List[np.ndarray]) -> Tuple[int, int]:
    """
    Estimate dominant periodicity along x and y from band transitions.
    Returns (repeat_x, repeat_y).
    """
    h, w = masks[0].shape
    lab = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks):
        lab[m] = i + 1

    tx = (lab[:, 1:] != lab[:, :-1]).astype(np.float32)  # x-edges
    ty = (lab[1:, :] != lab[:-1, :]).astype(np.float32)  # y-edges
    sx = tx.sum(axis=0)  # length w-1
    sy = ty.sum(axis=1)  # length h-1

    def _fft_peak(sig: np.ndarray) -> int:
        sig = sig - np.nan_to_num(sig.mean(), nan=0.0)
        if not np.any(np.isfinite(sig)) or np.allclose(sig, 0):
            return 0
        spec = np.abs(np.fft.rfft(sig))
        if spec.size:
            spec[0] = 0  # drop DC
        return int(np.argmax(spec)) if spec.size else 0

    # ✅ return the pair of peaks
    return _fft_peak(sx), _fft_peak(sy)


def _cluster_feats(img_rgb: np.ndarray, depth: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
    """Per-band descriptors: 32-bin grayscale hist + depth mean/std (len=34)."""
    gray = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.float32) / 255.0
    feats = []
    for m in masks:
        if m.sum() < 10:
            feats.append(np.zeros(34, dtype=np.float32))
            continue
        hist, _ = np.histogram(gray[m], bins=32, range=(0, 1), density=True)
        dvals = depth[m]
        feats.append(np.concatenate([hist.astype(np.float32),
                                     np.array([dvals.mean(), dvals.std()], dtype=np.float32)]))
    return feats


def run_gs_proxy_for_frame(image_path: str | Path, depth_cache_dir: str | Path) -> Dict[str, Any]:
    """
    Plug-compatible entry point (structured like your DINO Cluster baseline).

    Returns:
      rules: []               # keep for compatibility
      repeats: [rx, ry]       # ints, x then y
      motion: []              # keep for compatibility
      proxy_mask: (H,W) uint8 0/255 – largest depth band
      slot_masks: list[(H,W) bool] – masks for each depth band
      cluster_feats: list[np.ndarray len=34] – per-band descriptors
      avg_sim: float – quick scalar for logging
    """
    image_path = Path(image_path)
    depth_cache_dir = Path(depth_cache_dir)

    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))

    #depth = _load_depth_from_cache(image_path, depth_cache_dir)
    depth = _load_depth_from_cache(image_path, depth_cache_dir)
    # Ensure single-channel depth (sometimes MiDaS saves 3-channel pseudo-RGB depth)
    #if depth.ndim == 3:
    #    depth = depth[..., 0]
    depth = np.squeeze(depth)  # <-- add this
    if depth.ndim != 2:
        depth = depth[..., 0]  # final fallback
    masks, _ = _depth_bands(depth, k=4)
    proxy = _largest_band_mask(masks)
    rx, ry = _dominant_repeats(masks)
    feats = _cluster_feats(rgb, depth, masks)
    avg_sim = float(np.mean([np.dot(f, f) for f in feats])) if feats else 0.0

    return {
        "rules": [],
        "repeats": [int(rx), int(ry)],
        "motion": [],
        "proxy_mask": (proxy.astype(np.uint8) * 255).tolist() if proxy is not None else None,
        "slot_masks": [m.astype(bool).tolist() for m in masks],
        "cluster_feats": [f.astype(np.float32).tolist() for f in feats],
        "avg_sim": avg_sim,
    }
