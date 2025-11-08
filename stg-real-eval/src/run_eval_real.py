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
from stg_real_eval.metrics.temporal import ade_fde, edit_consistency_iou, replay_iou
from stg_real_eval.metrics.structural import delta_similarity, purity, facade_grid_score
from stg_real_eval.scripts.extract_features import extract_scene

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
    # Feature cache (dummy by default)
    extract_scene(scene.dataset, scene.scene_id, [str(f.image_path) for f in frames])
    preds = []
    t0 = time.time()
    # --- Direct call to native infer_image() ---
    for fr in frames:
        try:
            pred = infer_image(fr.image_path)  # returns dict with rules/depth/repeats/optionally motion
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Inference failed for {fr.image_path}: {exc}")
            pred = {"rules": [], "repeats": [0, 0], "depth": 0}
        preds.append(pred)
    runtime = time.time() - t0

    # Placeholder structural metrics (to be replaced with real data)
    feat_matrix = np.random.randn(len(frames), 128)
    pred_labels = np.arange(len(frames)) % 3
    gt_labels = np.arange(len(frames)) % 3
    dummy_mask = np.random.randint(0, 2, (64, 64))

    dsim = delta_similarity(feat_matrix)
    pur = purity(pred_labels, gt_labels)
    fgrid = facade_grid_score(dummy_mask)

    # Minimal temporal summaries (if we can make any)
    ade, fde = ade_fde([], [])  # left as NaN until tracker is plugged in
    rep_iou = np.nan
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
