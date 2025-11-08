import argparse
import datetime
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from metrics.metrics import (
    mdl_score,
    motion_error,
    persistence_score,
    regularity_error,
    rule_f1,
)


from utils.visualize_results import plot_summary_bars, make_overlays

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

TRACKER_PATH = (Path(__file__).resolve().parents[2] / "stg-stsg-model" / "src" / "experiment_tracker.py")


def _noop(*args, **kwargs):
    pass




def _create_run_folder_fallback(base_out="outputs/experiments", config_path=None, tag=None):
    os.makedirs(base_out, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    if tag:
        run_name += f"_{tag}"
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.yaml'))
    return run_dir, timestamp

if TRACKER_PATH.exists():
    spec = importlib.util.spec_from_file_location("stg_stsg_model_experiment_tracker", TRACKER_PATH)
    tracker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker_module)
    log_results = tracker_module.log_results
    append_global_log = tracker_module.append_global_log
    plot_progress = tracker_module.plot_progress
    create_run_folder = getattr(tracker_module, 'create_run_folder', _create_run_folder_fallback)
else:
    log_results = append_global_log = plot_progress = _noop
    create_run_folder = _create_run_folder_fallback



def evaluate_dataset(cfg):
    results = []
    dataset_dir = cfg.get('dataset')
    if not dataset_dir:
        raise ValueError('Dataset path must be provided via config or --dataset.')
    files = [f for f in os.listdir(dataset_dir) if f.endswith("_pred.json")]

    for f_pred in tqdm(files, desc="Evaluating"):
        import re
        # normalize names like scene_000_depth_pred.json → scene_000_gt.json
        f_gt = re.sub(r"(_(depth|mask))?_pred\.json$", "_gt.json", f_pred)
        with open(os.path.join(dataset_dir, f_pred), "r", encoding="utf-8") as f:
            pred = json.load(f)
        f_gt_path = os.path.join(dataset_dir, f_gt)
        if not os.path.exists(f_gt_path):
            shared_gt = os.path.join('datasets/synthetic_facades', f_gt)
            if os.path.exists(shared_gt):
                f_gt_path = shared_gt
            else:
                print(f'[warn] missing GT for {f_pred}, skipping.')
                continue
        with open(f_gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)

        res = {
            "scene": f_pred.replace("_pred.json", ""),
            "rule_f1": rule_f1(pred, gt),
            "reg_error": regularity_error(pred, gt),
            "mdl": mdl_score(pred),
            "persist": persistence_score(pred, gt),
            "motion_err": motion_error(pred, gt),
        }
        results.append(res)

    df = pd.DataFrame(results)
    outdir = cfg.get('output_dir', 'outputs/facades_demo')
    os.makedirs(outdir, exist_ok=True)
    results_path = os.path.join(outdir, "results.csv")
    df.to_csv(results_path, index=False)
    summary_path = os.path.join(outdir, "summary.png")
    plot_summary_bars(df, summary_path)

    print(df.describe())
    return results_path, summary_path, df


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/synthetic_facades.yaml')
    parser.add_argument('--dataset', default='datasets/synthetic_facades',
                        help='dataset folder containing *_pred.json and *_gt.json')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    if args.dataset:
        cfg['dataset'] = args.dataset
    if 'output_dir' not in cfg:
        cfg['output_dir'] = 'outputs/facades_demo'

    results_path, summary_path, _ = evaluate_dataset(cfg)

    run_base = os.path.join('outputs', 'experiments')
    os.makedirs(run_base, exist_ok=True)
    exp_tag = os.environ.get('EXP_TAG', 'eval')
    config_path = args.config if args.config and os.path.exists(args.config) else None
    run_dir, timestamp = create_run_folder(base_out=run_base, config_path=config_path, tag=exp_tag)
    print(f'[tracker] new run folder created: {run_dir}')

    log_results(run_dir, results_path, summary_path, config_path=config_path)
    make_overlays(args.dataset, run_dir, n=4)
    log_path = os.path.join(run_base, 'experiments_log.csv')
    append_global_log(log_path, timestamp, results_path)
    plot_progress(log_path)
    print(f'📊 Experiment logged: {run_dir}')


if __name__ == "__main__":
    _main()
