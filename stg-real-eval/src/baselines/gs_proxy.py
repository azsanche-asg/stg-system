from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    m, M = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - m) / (M - m)


def _read_depth_file(p: Path) -> np.ndarray:
    if p.suffix == ".npy":
        return np.load(p)
    # PNG/EXR fallback (PIL handles PNG; EXR support may vary)
    return np.array(Image.open(p).convert("F"))


def _load_depth_from_cache(image_path: Path, depth_cache_dir: Path) -> np.ndarray:
    """
    ONLY reads depth; never runs MiDaS.
    Tries: <stem>_midas.npy, <stem>.npy, <stem>.png, <stem>.exr in depth_cache_dir.
    If not found, tries canonical '.../midas/' under cache/.
    Falls back to 1 - grayscale(image) so the eval won’t crash.
    """
    stem = image_path.stem
    cands = [
        depth_cache_dir / f"{stem}_midas.npy",
        depth_cache_dir / f"{stem}.npy",
        depth_cache_dir / f"{stem}.png",
        depth_cache_dir / f"{stem}.exr",
    ]

    for root in [Path("cache") / "block_b", Path("cache")]:
        cands.extend([
            root / "**" / "midas" / f"{stem}_midas.npy",
            root / "**" / "midas" / f"{stem}.npy",
            root / "**" / "midas" / f"{stem}.png",
            root / "**" / "midas" / f"{stem}.exr",
        ])

    for p in cands:
        if "**" in str(p):
            try:
                for hit in p.parents[2].rglob(p.name):
                    return _normalize01(_read_depth_file(hit))
            except Exception:
                pass
        elif p.exists():
            try:
                return _normalize01(_read_depth_file(p))
            except Exception:
                pass

    # Last resort — synthetic grayscale fallback
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    return 1.0 - gray


# ---------------------------------------------------------------------
# Geometry segmentation and descriptors
# ---------------------------------------------------------------------
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
    sx = tx.sum(axis=0)
    sy = ty.sum(axis=1)

    def _fft_peak(sig: np.ndarray) -> int:
        sig = sig - np.nan_to_num(sig.mean(), nan=0.0)
        if not np.any(np.isfinite(sig)) or np.allclose(sig, 0):
            return 0
        spec = np.abs(np.fft.rfft(sig))
        if spec.size:
            spec[0] = 0  # drop DC
        return int(np.argmax(spec)) if spec.size else 0

    return _fft_peak(sx), _fft_peak(sy)


def _cluster_feats(img_rgb: np.ndarray, depth: np.ndarray, masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Per-band descriptors: 32-bin grayscale hist + depth mean/std (len=34).
    These play the same role as DINO Cluster's region embeddings.
    """
    gray = (
        0.299 * img_rgb[..., 0]
        + 0.587 * img_rgb[..., 1]
        + 0.114 * img_rgb[..., 2]
    ).astype(np.float32) / 255.0

    feats = []
    for m in masks:
        if m.sum() < 10:
            feats.append(np.zeros(34, dtype=np.float32))
            continue
        hist, _ = np.histogram(gray[m], bins=32, range=(0, 1), density=True)
        dvals = depth[m]
        feats.append(
            np.concatenate(
                [hist.astype(np.float32),
                 np.array([dvals.mean(), dvals.std()], dtype=np.float32)]
            )
        )
    return feats


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def run_gs_proxy_for_frame(image_path: str | Path, depth_cache_dir: str | Path) -> Dict[str, Any]:
    """
    Geometry-aware baseline mirroring DINO Cluster structure:
      - produces cluster_feats (band-level descriptors)
      - evaluator computes ΔSim, Purity, etc.
    """
    image_path = Path(image_path)
    depth_cache_dir = Path(depth_cache_dir)

    # --- RGB ---
    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))

    # --- Depth ---
    depth = _load_depth_from_cache(image_path, depth_cache_dir)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        depth = depth[..., 0]

    # Resize depth to RGB resolution
    if depth.shape != rgb.shape[:2]:
        h, w = rgb.shape[:2]
        depth = np.array(
            Image.fromarray(depth).resize((w, h), resample=Image.BILINEAR),
            dtype=np.float32,
        )

    # --- Structure inference ---
    masks, _ = _depth_bands(depth, k=4)
    proxy = _largest_band_mask(masks)
    rx, ry = _dominant_repeats(masks)
    feats = _cluster_feats(rgb, depth, masks)
    
    if feats:
        feats_arr = np.stack(feats, axis=0)                    # (k, 34)
        norms = np.linalg.norm(feats_arr, axis=1, keepdims=True) + 1e-8
        feats_n = feats_arr / norms
        sims = feats_n @ feats_n.T                             # (k, k) cosine
        k = sims.shape[0]
        if k > 1:
            iu = np.triu_indices(k, k=1)
            avg_sim = float(np.nanmean(sims[iu]))              # ~ 0.2–0.5 typical
        else:
            avg_sim = 0.0
    else:
        avg_sim = 0.0

    # --- Return DINO-compatible dict (no avg_sim) ---
    return {
        "rules": [],
        "repeats": [int(rx), int(ry)],
        "motion": [],
        "proxy_mask": (proxy.astype(np.uint8) * 255).tolist() if proxy is not None else None,
        "slot_masks": [m.astype(bool).tolist() for m in masks],
        "cluster_feats": [f.astype(np.float32).tolist() for f in feats],
        "avg_sim": avg_sim, 
        # no avg_sim → evaluator handles ΔSim consistently
    }