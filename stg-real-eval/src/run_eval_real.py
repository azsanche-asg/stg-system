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
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import yaml

import sys

THIS_FILE = Path(__file__).resolve()

if __package__:
    from .data import NuScenesMini, CityscapesSeq, CMPFacade
    from .scripts.extract_features import extract_scene
    from .metrics.temporal import ade_fde, replay_iou, edit_consistency_iou
    from .metrics.efficiency import footprint
else:  # Allows `python stg-real-eval/src/run_eval_real.py`
    pkg_root = THIS_FILE.parent
    if str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))
    from data import NuScenesMini, CityscapesSeq, CMPFacade
    from scripts.extract_features import extract_scene
    from metrics.temporal import ade_fde, replay_iou, edit_consistency_iou
    from metrics.efficiency import footprint

# Reuse model code
sys.path.append(str(THIS_FILE.parents[2] / "stg-stsg-model" / "src"))
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
    # Feature cache (dummy by default)
    extract_scene(scene.dataset, scene.scene_id, [str(f.image_path) for f in frames])
    preds = []
    t0 = time.time()
    for fr in frames:
        img = Image.open(fr.image_path).convert("RGB")
        pred = infer_image(img)  # returns dict with rules/depth/repeats/optionally motion
        preds.append(pred)
    runtime = time.time() - t0

    # Minimal temporal summaries (if we can make any)
    ade, fde = ade_fde([], [])  # left as NaN until tracker is plugged in
    rep_iou = np.nan
    edit_iou = np.nan

    out = {
        "dataset": scene.dataset,
        "scene_id": scene.scene_id,
        "num_frames": len(frames),
        "metrics": {
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


if __name__ == "__main__":
    main()
