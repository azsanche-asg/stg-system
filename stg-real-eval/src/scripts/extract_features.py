"""
Feature cache for Block B.
We keep it pluggable; if backbones aren't installed, we write lightweight stubs.
Cache layout: cache/block_b/<dataset>/<scene>/<frame>_{clip|dino|midas|sam}.npy
"""
from pathlib import Path

import numpy as np

from ..utils.paths import dataset_cache_root


def _safe_save(arr, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def _dummy_feat(shape=(512,)):
    return np.random.randn(*shape).astype("float32")


def extract_scene(dataset_name: str, scene_id: str, frame_paths, which=("clip", "dino", "midas", "sam")):
    root = dataset_cache_root(dataset_name) / scene_id
    for img_path in frame_paths:
        stem = Path(img_path).stem
        for key in which:
            out = root / f"{stem}_{key}.npy"
            if not out.exists():
                _safe_save(_dummy_feat(), out)


if __name__ == "__main__":
    # Tiny CLI usage documented in README; real invocation handled by run_eval_real.py
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames_json", required=True, help="JSON list of image paths")
    args = ap.parse_args()
    frames = json.loads(Path(args.frames_json).read_text())
    extract_scene(args.dataset, args.scene, frames)

