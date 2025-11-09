"""
Aggregates Block B evaluation metrics across datasets.
Scans results/block_b/*/*.json and prints average ± std for each metric.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
results_root = ROOT / "results" / "block_b"


def collect_jsons():
    files = list(results_root.glob("*/*.json"))
    grouped = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping invalid JSON: {f} ({exc})")
            continue
        ds = data.get("dataset", "unknown")
        grouped.setdefault(ds, []).append(data)
    return grouped


def summarize(grouped):
    for ds, entries in grouped.items():
        metrics = {}
        for e in entries:
            for k, v in e.get("metrics", {}).items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float)) and not np.isnan(subv):
                            metrics.setdefault(f"eff_{subk}", []).append(subv)
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    metrics.setdefault(k, []).append(v)
        print(f"\n### Dataset: {ds} ({len(entries)} scenes)")
        print("| Metric | Mean ± Std |")
        print("| :-- | --: |")
        for k, vals in metrics.items():
            arr = np.array(vals, dtype=float)
            print(f"| {k} | {arr.mean():.3f} ± {arr.std():.3f} |")
        print()


if __name__ == "__main__":
    if not results_root.exists():
        print("No results folder found at", results_root)
    else:
        grouped = collect_jsons()
        if not grouped:
            print("No result JSONs found under", results_root)
        else:
            summarize(grouped)
