import json
import os

import cv2
import matplotlib.pyplot as plt


def draw_grid(img, floors, repeats, color=(0, 255, 255)):
    height, width = img.shape[:2]
    for idx in range(1, max(int(floors), 1)):
        y = int(idx * height / max(floors, 1))
        cv2.line(img, (0, y), (width, y), color, 1)
    for idx in range(1, max(int(repeats), 1)):
        x = int(idx * width / max(repeats, 1))
        cv2.line(img, (x, 0), (x, height), color, 1)
    return img


def plot_summary_bars(df, out_path):
    plt.figure(figsize=(4, 3))
    plt.bar(
        ["Rule-F1", "Reg-Err", "MDL"],
        [df["rule_f1"].mean(), df["reg_error"].mean(), df["mdl"].mean()],
    )
    plt.title("Synthetic Evaluation Summary")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_overlays(dataset_dir, run_dir, n=4):
    examples_dir = os.path.join(run_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    preds = [fname for fname in os.listdir(dataset_dir) if fname.endswith("_pred.json")]
    preds.sort()
    preds = preds[:n]

    for fname_pred in preds:
        fname_gt = fname_pred.replace("_pred.json", "_gt.json")
        fname_img = fname_pred.replace("_pred.json", ".png")

        gt_path = os.path.join(dataset_dir, fname_gt)
        pred_path = os.path.join(dataset_dir, fname_pred)
        img_path = os.path.join(dataset_dir, fname_img)

        if not os.path.exists(img_path):
            alt_img_path = os.path.join('/content/stg-procedural-data/outputs/facades', fname_img)
            if os.path.exists(alt_img_path):
                img_path = alt_img_path
        if not (os.path.exists(gt_path) and os.path.exists(pred_path) and os.path.exists(img_path)):
            continue

        with open(gt_path, "r", encoding="utf-8") as f_gt:
            gt = json.load(f_gt)
        with open(pred_path, "r", encoding="utf-8") as f_pred:
            pr = json.load(f_pred)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_floors, gt_repeats = gt.get("repeats", [1, 1])
        pr_floors, pr_repeats = pr.get("repeats", [1, 1])

        img_gt = draw_grid(img.copy(), gt_floors, gt_repeats, color=(0, 255, 255))
        img_pr = draw_grid(img.copy(), pr_floors, pr_repeats, color=(255, 255, 0))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img_gt)
        axes[0].set_title(f"GT floors={gt_floors}, repeats={gt_repeats}")
        axes[1].imshow(img_pr)
        axes[1].set_title(f"PR floors={pr_floors}, repeats={pr_repeats}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_name = os.path.join(examples_dir, fname_pred.replace("_pred.json", ".png"))
        plt.savefig(out_name)
        plt.close()
