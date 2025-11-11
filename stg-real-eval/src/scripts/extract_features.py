"""
Feature cache for Block B.
Backbones are lightweight and lazily loaded (MiDaS, DINOv2, CLIP, SAM).
Cache layout: cache/block_b/<dataset>/<scene>/<frame>_{clip|dino|midas|sam}.npy
"""
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

try:
    import clip
except ImportError as exc:  # pragma: no cover
    clip = None

from ..utils.paths import dataset_cache_root

_midas = _dino = _clip = _clip_pre = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        _dino = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        _dino.eval()
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
    print("[INFO] SAM is disabled by default. Skipping SAM feature extraction.")
    return None


def _to_tensor(im: Image.Image):
    return _BASE_TRANSFORM(im).unsqueeze(0)


def extract_scene(dataset_name: str, scene_id: str, frame_paths, which=("clip", "dino", "midas")):
    env = os.environ.get("STG_FEATURES")
    if env:
        which = tuple([w.strip() for w in env.split(",") if w.strip()])
    device = _device()
    root = dataset_cache_root(dataset_name) / scene_id
    root.mkdir(parents=True, exist_ok=True)

    midas = _load_midas() if "midas" in which else None
    dino = _load_dino() if "dino" in which else None
    clip_model, clip_pre = _load_clip() if "clip" in which else (None, None)

    if midas is not None:
        midas = midas.to(device)
    if dino is not None:
        dino = dino.to(device)
    if clip_model is not None:
        clip_model = clip_model.to(device)

    if "sam" in which:
        _load_sam()

    for img_path in frame_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Skipping unreadable image: {img_path} ({exc})")
            continue
        stem = Path(img_path).stem
        tensor = _to_tensor(img).to(device)

        if "midas" in which and midas is not None:
            try:
                with torch.no_grad():
                    depth = midas(tensor).detach().cpu().numpy()
                np.save(root / f"{stem}_midas.npy", depth)
            except Exception as exc:
                print(f"[WARN] MiDaS failed on {img_path}: {exc}")

        if "dino" in which and dino is not None:
            try:
                resize_224 = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ]
                )
                im_t = resize_224(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = dino(im_t)
                feat = logits.detach().cpu().numpy().reshape(-1)
                np.save(root / f"{stem}_dino.npy", feat)
            except Exception as exc:
                print(f"[WARN] DINO failed on {img_path}: {exc}")

        if "clip" in which and clip_model is not None:
            try:
                with torch.no_grad():
                    im_proc = clip_pre(img).unsqueeze(0).to(device)
                    feat = clip_model.encode_image(im_proc).detach().cpu().numpy().reshape(-1)
                np.save(root / f"{stem}_clip.npy", feat)
            except Exception as exc:
                print(f"[WARN] CLIP failed on {img_path}: {exc}")

        if "sam" in which:
            print(f"[INFO] SAM skipped for {img_path} (disabled).")

    print(f"✅ Cached {len(frame_paths)} frames for {dataset_name}/{scene_id} using features: {which}")


def compute_midas_batch(frame_paths):
    """Compute MiDaS depth maps for a batch of image paths."""
    device = _device()
    model = _load_midas().to(device)
    outputs = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = _to_tensor(img).to(device)
            with torch.no_grad():
                depth = model(tensor).detach().cpu().numpy()
            if depth.ndim == 4:
                depth = depth[0]
            outputs.append(depth)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] compute_midas_batch failed for {path}: {exc}")
            outputs.append(None)
    return outputs


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
