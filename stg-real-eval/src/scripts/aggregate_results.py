"""
Aggregates Block B evaluation metrics across datasets.
Scans results/block_b/*/*.json and prints average ± std for each metric.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    all_dfs = []
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

        rows = []
        for k, vals in metrics.items():
            arr = np.array(vals, dtype=float)
            rows.append(
                {
                    "dataset": ds,
                    "metric": k,
                    "mean": arr.mean(),
                    "std": arr.std(),
                    "n_scenes": len(entries),
                }
            )

        df = pd.DataFrame(rows)
        all_dfs.append(df)

        print(f"\n### Dataset: {ds} ({len(entries)} scenes)")
        if df.empty:
            print("No numeric metrics to summarize.")
        else:
            print(
                df.to_string(
                    index=False,
                    formatters={
                        "mean": lambda x: f"{x:.3f}",
                        "std": lambda x: f"{x:.3f}",
                    },
                )
            )

        csv_path = results_root / f"{ds}_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")

        latex_path = results_root / f"{ds}_summary.tex"
        with open(latex_path, "w", encoding="utf-8") as handle:
            handle.write(df.to_latex(index=False, float_format="%.3f", caption=f"Summary for {ds}"))
        print(f"LaTeX table saved to {latex_path}")

    merged = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    merged_path = results_root / "all_datasets_summary.csv"
    merged.to_csv(merged_path, index=False)
    print("\nGlobal CSV saved to", merged_path)


if __name__ == "__main__":
    if not results_root.exists():
        print("No results folder found at", results_root)
    else:
        grouped = collect_jsons()
        if not grouped:
            print("No result JSONs found under", results_root)
        else:
            summarize(grouped)
