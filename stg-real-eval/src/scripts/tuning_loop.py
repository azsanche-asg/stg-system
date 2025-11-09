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
    metrics = {"ΔSim": [], "Purity": [], "ADE": [], "FDE": [], "ReplayIoU": []}
    for f in all_jsons:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to parse {f}: {exc}")
            continue
        m = data.get("metrics", {})
        metrics["ΔSim"].append(m.get("delta_similarity", np.nan))
        metrics["Purity"].append(m.get("purity", np.nan))
        metrics["ADE"].append(m.get("ade", np.nan))
        metrics["FDE"].append(m.get("fde", np.nan))
        metrics["ReplayIoU"].append(m.get("replay_iou", np.nan))
    avg = {k: float(np.nanmean(v)) if v else np.nan for k, v in metrics.items()}
    avg["Score"] = avg["ΔSim"] + avg["Purity"] + avg["ReplayIoU"] - 0.5 * (avg["ADE"] + avg["FDE"])
    return avg


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

    avg = parse_results()
    avg.update({"features": features, "lambda": lam, "beam": beam})
    log_rows.append(avg)

df = pd.DataFrame(log_rows)
log_path = results_root / "tuning_log.csv"
df.to_csv(log_path, index=False)
print("\n=== Tuning summary ===")
if not df.empty:
    print(df.sort_values("Score", ascending=False).to_string(index=False, float_format="%.3f"))
else:
    print("No successful runs recorded.")
print("\nCSV written to", log_path)
