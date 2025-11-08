import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.io_utils import save_depth, save_image, save_json, save_mask


def generate_street(width=256, height=256, lanes=2, cars_per_lane=4):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask = np.zeros((height, width), dtype=np.int32)
    depth = np.ones((height, width), dtype=np.float32)
    rule_tree = {"Z": "root", "rules": [], "repeats": [], "depth": 0, "persist_ids": [], "motion": []}

    lane_padding = max(height // 10, 12)
    road_top = lane_padding
    road_bottom = height - lane_padding
    cv2.rectangle(img, (0, road_top), (width, road_bottom), (180, 180, 180), -1)
    rule_tree["rules"].append(f"Split_y_{lanes}")

    lane_height = max((road_bottom - road_top) // max(lanes, 1), 1)
    label_id = 1

    for lane_idx in range(lanes):
        y0 = road_top + lane_idx * lane_height
        y1 = road_bottom if lane_idx == lanes - 1 else road_top + (lane_idx + 1) * lane_height
        center_line = (y0 + y1) // 2
        cv2.line(img, (0, center_line), (width, center_line), (250, 250, 100), 1)

        car_width = max(width // (cars_per_lane * 3), 12)
        for car_idx in range(cars_per_lane):
            x_center = int((car_idx + 1) * width / (cars_per_lane + 1))
            x0 = max(x_center - car_width // 2, 0)
            x1 = min(x_center + car_width // 2, width)
            car_inset = max((y1 - y0) // 8, 2)
            cv2.rectangle(img, (x0, y0 + car_inset), (x1, y1 - car_inset), (60, 60, 200), -1)
            mask[y0:y1, x0:x1] = label_id
            depth[y0:y1, x0:x1] = 1.0 + 0.05 * lane_idx
            label_id += 1

        rule_tree["rules"].append(f"Repeat_x_{cars_per_lane}")

    rule_tree["repeats"] = [lanes, cars_per_lane]
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

    for idx in tqdm(range(args.n), desc="Generating streets"):
        lanes = int(np.random.randint(1, 4))
        cars = int(np.random.randint(2, 6))
        img, depth, mask, gt = generate_street(
            width=args.width,
            height=args.height,
            lanes=lanes,
            cars_per_lane=cars,
        )

        prefix = os.path.join(args.out, f"street_{idx:03d}")
        save_image(img, f"{prefix}.png")
        save_depth(depth, f"{prefix}_depth.npy")
        save_mask(mask, f"{prefix}_mask.npy")
        save_json(gt, f"{prefix}_gt.json")


if __name__ == "__main__":
    main()
