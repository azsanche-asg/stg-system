import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def autocorr(signal):
    signal = signal - np.mean(signal)
    result = np.correlate(signal, signal, mode="full")
    result = result[result.size // 2 :]
    return result / (np.max(result) + 1e-8)


def run_seg_repeat(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in tqdm(os.listdir(img_dir)):
        if not fname.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(img_dir, fname), 0)
        proj_x = img.mean(axis=0)
        proj_y = img.mean(axis=1)
        corr_x = autocorr(proj_x)
        corr_y = autocorr(proj_y)
        rep_x = int(np.argmax(corr_x[1:10]) + 1)
        rep_y = int(np.argmax(corr_y[1:10]) + 1)
        pred = {
            "rules": [f"Repeat_x_{rep_x}", f"Repeat_y_{rep_y}"],
            "repeats": [rep_x, rep_y],
            "depth": 1,
            "persist_ids": [],
            "motion": [],
        }
        base_name = fname.replace('_mask.png', '_pred.json') if '_mask.png' in fname else fname.replace('.png', '_pred.json')  # ensure naming consistency
        out_path = os.path.join(out_dir, base_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pred, f)
