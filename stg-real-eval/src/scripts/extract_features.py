"""
Feature cache for Block B.
Backbones are lightweight and lazily loaded (MiDaS, DINOv2, CLIP, SAM).
Cache layout: cache/block_b/<dataset>/<scene>/<frame>_{clip|dino|midas|sam}.npy
"""
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

try:
    import clip
except ImportError as exc:  # pragma: no cover
    clip = None

from ..utils.paths import dataset_cache_root

_midas = _dino = _clip = _clip_pre = _sam = None
_BASE_TRANSFORM = Compose(
    [
        Resize(384),
        CenterCrop(384),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def _load_midas():
    global _midas
    if _midas is None:
        _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()
    return _midas


def _load_dino():
    global _dino
    if _dino is None:
        _dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8").eval()
    return _dino


def _load_clip():
    if clip is None:
        raise ImportError("clip package is required but not installed. Install via pip.")
    global _clip, _clip_pre
    if _clip is None:
        _clip, _clip_pre = clip.load("ViT-B/16", device="cpu")
        _clip.eval()
    return _clip, _clip_pre


def _load_sam():
    global _sam
    if _sam is None:
        _sam = torch.hub.load("facebookresearch/segment-anything", "vit_h").eval()
    return _sam


def _to_tensor(im: Image.Image):
    return _BASE_TRANSFORM(im).unsqueeze(0)


def extract_scene(dataset_name: str, scene_id: str, frame_paths, which=("clip", "dino", "midas", "sam")):
    env = os.environ.get("STG_FEATURES")
    if env:
        which = tuple([w.strip() for w in env.split(",") if w.strip()])
    root = dataset_cache_root(dataset_name) / scene_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root.mkdir(parents=True, exist_ok=True)

    midas = _load_midas().to(device)
    dino = _load_dino().to(device)
    clip_model, clip_pre = _load_clip()
    clip_model = clip_model.to(device)
    sam = _load_sam().to(device)

    for img_path in frame_paths:
        img = Image.open(img_path).convert("RGB")
        stem = Path(img_path).stem
        tensor = _to_tensor(img).to(device)

        with torch.no_grad():
            if "midas" in which:
                depth = midas(tensor).detach().cpu().numpy()
                np.save(root / f"{stem}_midas.npy", depth)

            if "dino" in which:
                feat = dino(tensor).mean(dim=[2, 3]).detach().cpu().numpy()
                np.save(root / f"{stem}_dino.npy", feat)

            if "clip" in which:
                im_proc = clip_pre(img).unsqueeze(0).to(device)
                feat = clip_model.encode_image(im_proc).cpu().numpy()
                np.save(root / f"{stem}_clip.npy", feat)

            if "sam" in which:
                masks = sam(tensor).cpu().numpy()
                np.save(root / f"{stem}_sam.npy", masks)

    print(f"✅ Features cached for {len(frame_paths)} frames → {root}")


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
