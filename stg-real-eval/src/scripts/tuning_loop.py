import itertools
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
results_root = ROOT / "results"
stsg_cfg_path = ROOT / "stg-stsg-model" / "configs" / "v1_facades.yaml"
cmp_cfg = ROOT / "stg-real-eval" / "configs" / "block_b_cmp.yaml"
city_cfg = ROOT / "stg-real-eval" / "configs" / "block_b_cityscapes.yaml"
nusc_cfg = ROOT / "stg-real-eval" / "configs" / "block_b_nuscenes.yaml"

feature_sets = ["clip", "clip,dino", "clip,dino,midas"]
lambdas = [0.1, 0.5, 1.0, 2.0]
beams = [3, 5, 7]

combos = random.sample(list(itertools.product(feature_sets, lambdas, beams)), k=min(10, len(feature_sets) * len(lambdas) * len(beams)))

log_rows = []


def run_eval(cfg_path):
    cmd = ["python", "stg-real-eval/src/run_eval_real.py", "--config", str(cfg_path)]
    print("Running", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_results():
    all_jsons = list(results_root.glob("block_b/*/*.json"))
    metrics_by_ds = {}
    for f in all_jsons:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to parse {f}: {exc}")
            continue
        ds = data.get("dataset", "unknown")
        m = data.get("metrics", {}) or {}
        if ds == "cmp-facade":
            m.pop("ade", None)
            m.pop("fde", None)
        metrics_by_ds.setdefault(
            ds, {"ΔSim": [], "Purity": [], "ADE": [], "FDE": [], "ReplayIoU": []}
        )
        metrics_by_ds[ds]["ΔSim"].append(m.get("delta_similarity", np.nan))
        metrics_by_ds[ds]["Purity"].append(m.get("purity", np.nan))
        metrics_by_ds[ds]["ADE"].append(m.get("ade", np.nan))
        metrics_by_ds[ds]["FDE"].append(m.get("fde", np.nan))
        metrics_by_ds[ds]["ReplayIoU"].append(m.get("replay_iou", np.nan))
    rows = []
    for ds, metrics in metrics_by_ds.items():
        avg = {}
        for k, vals in metrics.items():
            arr = np.array([x for x in vals if not np.isnan(x)])
            avg[k] = float(arr.mean()) if len(arr) > 0 else np.nan
        ade_norm = avg["ADE"] / 10.0 if not np.isnan(avg["ADE"]) else 0
        fde_norm = avg["FDE"] / 10.0 if not np.isnan(avg["FDE"]) else 0
        avg["Score"] = avg["ΔSim"] + avg["Purity"] + avg["ReplayIoU"] - 0.5 * (ade_norm + fde_norm)
        avg["dataset"] = ds
        rows.append(avg)
    return rows


for features, lam, beam in combos:
    os.environ["STG_FEATURES"] = features

    cfg = yaml.safe_load(stsg_cfg_path.read_text())
    cfg.setdefault("search", {})["beam_width"] = beam
    cfg.setdefault("weights", {})
    cfg["weights"]["lambda_split"] = lam
    cfg["weights"]["lambda_repeat"] = lam
    cfg["weights"]["lambda_render"] = lam
    stsg_cfg_path.write_text(yaml.safe_dump(cfg))

    print(f"\n=== Running combo ===\nFeatures={features} | λ={lam} | Beam={beam}\n")
    for cfg_file in [cmp_cfg, city_cfg, nusc_cfg]:
        try:
            run_eval(cfg_file)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Run failed for {cfg_file}: {exc}")

    avg_rows = parse_results()
    for row in avg_rows:
        row.update({"features": features, "lambda": lam, "beam": beam})
        log_rows.append(row)

df = pd.DataFrame(log_rows)
log_path = results_root / "tuning_log.csv"
df.to_csv(log_path, index=False)
print("\n=== Global tuning summary (all datasets combined) ===")
if not df.empty:
    print(df.sort_values("Score", ascending=False).to_string(index=False, float_format="%.3f"))
else:
    print("No successful runs recorded.")
print("\nCSV written to", log_path)

if not df.empty:
    print("\n=== Best configurations per dataset ===")
    for ds in df["dataset"].dropna().unique():
        df_ds = df[df["dataset"] == ds]
        best = df_ds.loc[df_ds["Score"].idxmax()]
        print(f"\n[{ds}]")
        print(best.to_string())

    ds_groups = df.groupby("dataset", dropna=True)
    ds_scores = []
    for ds, sub in ds_groups:
        ds_scores.append(sub["Score"].mean(skipna=True))
    if ds_scores:
        global_mean = np.mean(ds_scores)
        print(f"\n=== Global average score across datasets: {global_mean:.3f} ===")
    else:
        print("\n=== No valid dataset scores to average ===")
