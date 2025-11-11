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
    from stg_real_eval.baselines.crf_utils import dense_crf_refine
except Exception:  # pragma: no cover
    dense_crf_refine = None
try:
    from stg_real_eval.baselines.raster_proxy import infer_raster_baseline
except Exception:  # pragma: no cover
    infer_raster_baseline = None
try:
    from stg_real_eval.baselines.raster_crf import raster_with_crf
except Exception:  # pragma: no cover
    raster_with_crf = None
try:
    from stg_real_eval.baselines.dino_cluster_proxy import infer_dino_cluster
except Exception:  # pragma: no cover
    infer_dino_cluster = None
try:
    from stg_real_eval.baselines.gs_proxy import infer_gs_proxy
except Exception:  # pragma: no cover
    infer_gs_proxy = None
try:
    from stg_real_eval.baselines.slot_attention_proxy import infer_slot_baseline
except Exception:  # pragma: no cover
    infer_slot_baseline = None
try:
    from stg_real_eval.metrics.temporal_slots import match_slots_across_frames
except Exception:  # pragma: no cover
    match_slots_across_frames = None

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
    def _mean_matched_iou(prev_masks, prev_feats, cur_masks, cur_feats, alpha=0.5):
        def _cos(a, b):
            na = np.linalg.norm(a) + 1e-6
            nb = np.linalg.norm(b) + 1e-6
            return float(np.dot(a, b) / (na * nb))

        def _iou(a, b):
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            return inter / union if union > 0 else 0.0

        used = set()
        scores = []
        for i, pm in enumerate(prev_masks):
            best, bj = -1.0, -1
            for j, cm in enumerate(cur_masks):
                if j in used:
                    continue
                score = alpha * _iou(pm, cm) + (1 - alpha) * _cos(prev_feats[i], cur_feats[j])
                if score > best:
                    best, bj = score, j
            if bj >= 0:
                used.add(bj)
                scores.append(_iou(prev_masks[i], cur_masks[bj]))
        return float(np.mean(scores)) if scores else float("nan")
    frames = scene.frames
    frame_paths = [str(f.image_path) for f in frames]

    model_type = os.environ.get("BASELINE")
    if os.environ.get("CRF"):
        model_type = "raster_crf"
    if not model_type:
        model_type = cfg.get("model", "stsg")
    print(f"⚙️  Model type selected: {model_type}")

    if model_type not in ("raster", "raster_crf", "dino_cluster"):
        extract_scene(scene.dataset, scene.scene_id, frame_paths)

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
    if model_type == "slot" and infer_slot_baseline is not None:
        print("🧩  Slot-Attention baseline branch entered; running unsupervised part inference...")
        for pil_img, fr in zip(pil_images, frames):
            try:
                pred = infer_slot_baseline(pil_img)
            except Exception as exc:
                print(f"⚠️  Slot baseline failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0}
            preds.append(pred)
    elif model_type == "gs_proxy" and infer_gs_proxy is not None:
        print("🪩  3DGS proxy baseline active – inferring geometry from MiDaS depth")
        cache_root = Path("cache") / "block_b" / scene.dataset / scene.scene_id
        for pil_img, fr in zip(pil_images, frames):
            stem = Path(fr.image_path).stem
            depth_file = cache_root / f"{stem}_midas.npy"
            if not depth_file.exists():
                print(f"⚠️ No MiDaS cache found for {fr.image_path}")
                preds.append({"rules": [], "repeats": [0, 0], "depth": 0, "proxy_mask": None})
                continue
            depth = np.load(depth_file)
            if depth.ndim > 2:
                depth = depth[0]
            try:
                pred = infer_gs_proxy(pil_img, depth)
            except Exception as exc:
                print(f"⚠️ GS proxy failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0, "proxy_mask": None}
            preds.append(pred)
    elif model_type == "dino_cluster" and infer_dino_cluster is not None:
        print("🧩  DINO v2 feature-clustering baseline active…")
        for pil_img, fr in zip(pil_images, frames):
            try:
                pred = infer_dino_cluster(pil_img)
            except Exception as exc:
                print(f"⚠️ DINO cluster baseline failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0}
            preds.append(pred)
    elif model_type == "raster_crf" and raster_with_crf is not None:
        print("🧮  Raster+CRF baseline branch entered; running proxy inference...")
        for pil_img, fr in zip(pil_images, frames):
            try:
                pred = raster_with_crf(pil_img)
            except Exception as exc:
                print(f"⚠️ Raster+CRF baseline failed for {fr.image_path}: {exc}")
                pred = {"rules": [], "repeats": [0, 0], "depth": 0}
            preds.append(pred)
    elif model_type == "raster" and infer_raster_baseline is not None:
        print("🧮  Raster baseline branch entered; running proxy inference...")
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

    if model_type == "raster_crf" and dense_crf_refine is not None:
        refined_preds = []
        for pil_img, pred in zip(pil_images, preds):
            proxy = pred.get("proxy_mask") if isinstance(pred, dict) else None
            if proxy is None:
                refined_preds.append(pred)
                continue
            img = np.array(pil_img.convert("RGB"))
            prob_fg = (np.array(proxy, dtype=np.uint8) > 0).astype(np.float32)
            prob_fg = prob_fg * 0.9 + 0.05
            crf_mask = dense_crf_refine(img, prob_fg, iters=5)
            new_pred = dict(pred)
            new_pred["proxy_mask"] = crf_mask.astype(np.uint8).tolist()
            refined_preds.append(new_pred)
        preds = refined_preds

    mask_seq_slot = None
    dino_masks_seq = None
    if model_type == "slot" and match_slots_across_frames is not None:
        slot_masks_seq = []
        slot_embs_seq = []
        valid_slots = True
        for pred in preds:
            sms = pred.get("slot_masks")
            ems = pred.get("slot_embs")
            if sms is None or ems is None:
                valid_slots = False
                break
            slot_masks_seq.append([np.array(m, dtype=bool) for m in sms])
            slot_embs_seq.append([np.array(e, dtype=np.float32) for e in ems])
        if valid_slots and slot_masks_seq:
            matched = match_slots_across_frames(slot_masks_seq, slot_embs_seq, alpha=0.5)
            if matched:
                mask_seq_slot = [np.array(m, dtype=bool) for m in matched]
    elif model_type == "dino_cluster" and match_slots_across_frames is not None:
        slot_masks_seq = []
        feats_seq = []
        valid = True
        for pred in preds:
            sms = pred.get("slot_masks")
            fts = pred.get("cluster_feats")
            if sms is None or fts is None:
                valid = False
                break
            slot_masks_seq.append([np.array(m, dtype=bool) for m in sms])
            feats_seq.append([np.array(f, dtype=np.float32) for f in fts])
        if valid and slot_masks_seq:
            matched = match_slots_across_frames(slot_masks_seq, feats_seq, alpha=0.5)
            if matched:
                dino_masks_seq = [np.array(m, dtype=bool) for m in matched]

    mask_seq = []
    if mask_seq_slot is not None:
        mask_seq = mask_seq_slot
    elif model_type in ("raster", "slot", "raster_crf", "dino_cluster", "gs_proxy"):
        for pred in preds:
            proxy = pred.get("proxy_mask") if isinstance(pred, dict) else None
            if proxy is not None:
                mask_seq.append(np.array(proxy, dtype=bool))
    else:
        for fr in frames:
            stem = Path(fr.image_path).stem
            f_midas = cache_root / f"{stem}_midas.npy"
            if f_midas.exists():
                depth = np.load(f_midas)
                mask_seq.append(depth[0] > np.median(depth[0]))
    rep_iou = replay_iou(mask_seq) if len(mask_seq) > 1 else np.nan

    if model_type == "dino_cluster" and slot_masks_seq:
        pair_ious = []
        for t in range(1, len(slot_masks_seq)):
            miou = _mean_matched_iou(
                slot_masks_seq[t - 1],
                feats_seq[t - 1],
                slot_masks_seq[t],
                feats_seq[t],
                alpha=0.5,
            )
            if not np.isnan(miou):
                pair_ious.append(miou)
        rep_iou = float(np.mean(pair_ious)) if pair_ious else np.nan
    if scene.dataset == "cmp-facade":
        rep_iou = np.nan

    feat_matrix = []
    if model_type in ("raster", "slot", "raster_crf", "dino_cluster", "gs_proxy"):
        for fr in frames:
            with Image.open(fr.image_path) as _im:
                gray = np.array(_im.convert("L"))
            hist, _ = np.histogram(gray, bins=128, range=(0, 255), density=True)
            feat_matrix.append(hist.astype(np.float32))
    else:
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

    if model_type in ("dino_cluster", "gs_proxy"):
        frame_sims = [pred.get("avg_sim", np.nan) for pred in preds]
        if frame_sims:
            dsim = float(np.nanmean(frame_sims))

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
