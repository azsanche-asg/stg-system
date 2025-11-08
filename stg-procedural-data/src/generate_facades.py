import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.io_utils import save_depth, save_image, save_json, save_mask


def generate_facade(width=256, height=256, floors=3, windows_per_floor=5):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask = np.zeros((height, width), dtype=np.int32)
    depth = np.ones((height, width), dtype=np.float32)
    rule_tree = {"Z": "root", "rules": [], "repeats": [], "depth": 0, "persist_ids": [], "motion": []}

    floor_height = height // max(floors, 1)
    window_width = width // max(windows_per_floor, 1)

    label_id = 1
    for floor_idx in range(floors):
        y0 = floor_idx * floor_height
        y1 = height if floor_idx == floors - 1 else (floor_idx + 1) * floor_height
        cv2.rectangle(img, (0, y0), (width, y1), (200, 200, 200), -1)
        rule_tree["rules"].append(f"Split_y_{floor_idx}")

        for window_idx in range(windows_per_floor):
            x0 = window_idx * window_width
            x1 = width if window_idx == windows_per_floor - 1 else (window_idx + 1) * window_width
            inset = max(min(window_width, floor_height) // 10, 3)
            cv2.rectangle(img, (x0 + inset, y0 + inset), (x1 - inset, y1 - inset), (100, 100, 250), -1)
            mask[y0:y1, x0:x1] = label_id
            depth[y0:y1, x0:x1] = 1.0 + 0.05 * floor_idx
            label_id += 1
        rule_tree["rules"].append(f"Repeat_x_{windows_per_floor}")

    rule_tree["repeats"] = [floors, windows_per_floor]
    rule_tree["depth"] = 2
    return img, depth, mask, rule_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for idx in tqdm(range(args.n), desc="Generating facades"):
        floors = int(np.random.randint(2, 6))
        windows = int(np.random.randint(3, 8))
        img, depth, mask, gt = generate_facade(
            width=args.width,
            height=args.height,
            floors=floors,
            windows_per_floor=windows,
        )

        prefix = os.path.join(args.out, f"scene_{idx:03d}")
        save_image(img, f"{prefix}.png")
        save_depth(depth, f"{prefix}_depth.npy")
        save_mask(mask, f"{prefix}_mask.npy")
        save_json(gt, f"{prefix}_gt.json")


if __name__ == "__main__":
    main()
