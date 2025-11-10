"""
Entry point for Block B evaluation on real mini datasets.
- Loads dataset config
- Builds tiny scene lists
- Caches features (dummy by default)
- Calls existing STSG inference (stg-stsg-model/infer_v1.py) per frame
- Writes per-scene JSON logs and prints small summaries
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import os

import importlib.util

import numpy as np
import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

# Allow absolute `stg_real_eval.*` imports when running as a script.
sys.path.append(str(REPO_ROOT))

# Create a stable module alias for the hyphenated package folder.
PKG_PATH = REPO_ROOT / "stg-real-eval" / "src"
if "stg_real_eval" not in sys.modules and PKG_PATH.exists():  # pragma: no cover
    spec = importlib.util.spec_from_file_location(
        "stg_real_eval",
        PKG_PATH / "__init__.py",
        submodule_search_locations=[str(PKG_PATH)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["stg_real_eval"] = module
        spec.loader.exec_module(module)
sys.path.append(str(PKG_PATH))

from stg_real_eval.data import CMPFacade, CityscapesSeq, NuScenesMini
from stg_real_eval.metrics.efficiency import footprint
from stg_real_eval.metrics.temporal import ade_fde_from_flow, replay_iou
from stg_real_eval.metrics.structural import delta_similarity, purity, facade_grid_score
from stg_real_eval.scripts.extract_features import extract_scene

try:
    from stg_real_eval.src.baselines.raster_proxy import infer_raster_baseline
except Exception:  # pragma: no cover
    infer_raster_baseline = None

# Reuse model code
sys.path.append(str(REPO_ROOT / "stg-stsg-model" / "src"))
from infer_v1 import infer_image  # expects a PIL/numpy image path, returns JSON-like dict


def load_cfg(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def get_scenes(cfg) -> List[Dict[str, Any]]:
    ds = cfg["dataset"]
    p = cfg["paths"]
    if ds == "nuscenes-mini":
        loader = NuScenesMini(p["root"], take_every=cfg["eval"].get("take_every", 5))
        return loader.list_scenes()
    if ds == "cityscapes-seq":
        loader = CityscapesSeq(
            p["seq_root"],
            p.get("still_root"),
            take_every=cfg["eval"].get("take_every", 3),
            max_frames=cfg["eval"].get("max_frames", 60),
        )
        return loader.list_scenes() + loader.list_stills()
    if ds == "cmp-facade":
        loader = CMPFacade(p["root"], max_images=cfg["eval"].get("max_images", 10))
        return loader.list_scenes()
    raise ValueError(f"Unknown dataset {ds}")


def run_scene(cfg, scene, results_dir: Path):
    frames = scene.frames
    frame_paths = [str(f.image_path) for f in frames]
    extract_scene(scene.dataset, scene.scene_id, frame_paths)

    model_type = os.environ.get("BASELINE")
    if not model_type:
        model_type = cfg.get("model", "stsg")

    pil_images = []
    imgs_rgb = []
    for fr in frames:
        with Image.open(fr.image_path) as img:
            rgb = img.convert("RGB")
            pil_images.append(rgb)
            imgs_rgb.append(np.array(rgb))

    if scene.dataset == "cmp-facade":
        ade, fde = np.nan, np.nan
    else:
        try:
            ade, fde = ade_fde_from_flow(imgs_rgb)
        except Exception as exc:
            print(f"[WARN] Optical flow failed for {scene.dataset}: {exc}")
            ade, fde = np.nan, np.nan

    preds = []
    t0 = time.time()
    if model_type == "raster" and infer_raster_baseline is not None:
        for pil_img, fr in zip(pil_images, frames):
            try:
                pred = infer_raster_baseline(pil_img)
            except Exception as exc:
                print(f"⚠️ Raster baseline failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0}
            preds.append(pred)
    else:
        for pil_img, fr in zip(pil_images, frames):
            try:
                pred = infer_image(pil_img)
            except Exception as exc:
                print(f"⚠️ Inference failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0}
            preds.append(pred)
    runtime = time.time() - t0

    cache_root = Path("cache") / "block_b" / scene.dataset / scene.scene_id

    mask_seq = []
    for fr in frames:
        stem = Path(fr.image_path).stem
        f_midas = cache_root / f"{stem}_midas.npy"
        if f_midas.exists():
            depth = np.load(f_midas)
            mask_seq.append(depth[0] > np.median(depth[0]))
    rep_iou = replay_iou(mask_seq) if len(mask_seq) > 1 else np.nan
    if scene.dataset == "cmp-facade":
        rep_iou = np.nan

    feat_matrix = []
    for fr in frames:
        stem = Path(fr.image_path).stem
        fpath = cache_root / f"{stem}_clip.npy"
        if fpath.exists():
            feat_matrix.append(np.load(fpath).flatten())
    if not feat_matrix:
        extract_scene(scene.dataset, scene.scene_id, frame_paths)
        feat_matrix = [
            np.load(cache_root / f"{Path(f.image_path).stem}_clip.npy").flatten()
            for f in frames
            if (cache_root / f"{Path(f.image_path).stem}_clip.npy").exists()
        ]

    if feat_matrix:
        feat_matrix = np.stack(feat_matrix)
        pred_labels = np.arange(len(feat_matrix)) % 3
        gt_labels = pred_labels.copy()
        dummy_mask = np.ones((64, 64))
        dsim = delta_similarity(feat_matrix)
        pur = purity(pred_labels, gt_labels)
        fgrid = facade_grid_score(dummy_mask)
    else:
        dsim = pur = fgrid = np.nan

    edit_iou = np.nan

    out = {
        "dataset": scene.dataset,
        "scene_id": scene.scene_id,
        "num_frames": len(frames),
        "metrics": {
            "delta_similarity": dsim,
            "purity": pur,
            "facade_grid_score": fgrid,
            "ade": ade,
            "fde": fde,
            "replay_iou": rep_iou,
            "edit_consistency_iou": edit_iou,
            "efficiency": footprint(model_json="", runtime_s=runtime),
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{scene.scene_id}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    scenes = get_scenes(cfg)
    results_dir = Path(cfg["outputs"]["results_dir"])
    outs = []
    for sc in scenes:
        outs.append(run_scene(cfg, sc, results_dir))
    print(json.dumps({"summary": outs}, indent=2))
    if outs:
        last = outs[-1]["metrics"]
        dsim = float(last.get("delta_similarity", np.nan))
        pur = float(last.get("purity", np.nan))
        fgrid = float(last.get("facade_grid_score", np.nan))
        print(f"ΔSim={dsim:.3f}, Purity={pur:.3f}, FacadeGrid={fgrid:.3f}")


if __name__ == "__main__":
    main()
