import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Rule-F1", "Reg-Err", "MDL", "Persistence"],
        [
            df["rule_f1"].mean(),
            df["reg_error"].mean(),
            df["mdl"].mean(),
            df["persist"].mean(),
        ],
    )
    plt.title("Synthetic Evaluation Summary")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "summary.png"))


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_summary(args.csv, args.out)


if __name__ == "__main__":
    _main()
