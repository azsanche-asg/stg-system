import os

import cv2
import json
import numpy as np
from tqdm import tqdm

from utils.io_utils import save_depth, save_image, save_json, save_mask
from src.generate_facades import generate_facade


def generate_temporal_facades(n=5, frames=4, out="outputs/temporal_facades"):
    """Create n short façade clips (each with 'frames' images) and small horizontal motion."""
    os.makedirs(out, exist_ok=True)
    for idx in tqdm(range(n), desc="Generating temporal facades"):
        floors = np.random.randint(2, 5)
        windows = np.random.randint(4, 8)
        img0, depth0, mask0, gt0 = generate_facade(floors=floors, windows_per_floor=windows)
        height, width = img0.shape[:2]
        ids = np.unique(mask0)[1:]
        persist_ids = [int(x) for x in ids]
        dx_per_frame = 2  # horizontal displacement per frame

        for t in range(frames):
            dx = t * dx_per_frame
            img = np.ones_like(img0) * 255
            mask = np.zeros_like(mask0)
            for obj_id in ids:
                ys, xs = np.where(mask0 == obj_id)
                xs_new = np.clip(xs + dx, 0, width - 1)
                img[ys, xs_new] = img0[ys, xs]
                mask[ys, xs_new] = obj_id

            prefix = os.path.join(out, f"scene_{idx:03d}_t{t}")
            save_image(img, prefix + ".png")
            save_depth(depth0, prefix + "_depth.npy")
            save_mask(mask, prefix + "_mask.npy")

            gt = dict(gt0)
            gt["persist_ids"] = persist_ids
            gt["motion"] = {str(int(obj_id)): [int(dx_per_frame), 0] for obj_id in ids}
            save_json(gt, prefix + "_gt.json")

    eval_dataset = os.path.abspath(os.path.join('..', 'stg-synthetic-eval', 'datasets', 'temporal_facades'))
    os.makedirs(eval_dataset, exist_ok=True)
    import shutil
    for fname in os.listdir(out):
        src = os.path.join(out, fname)
        dst = os.path.join(eval_dataset, fname)
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            print(f'[warn] could not copy {fname} -> eval dataset: {exc}')
    print(f'✅ Generated {n} temporal façade clips × {frames} frames each in {out}')
    print(f'🗂️  Synced with evaluation dataset: {eval_dataset}')


if __name__ == "__main__":
    generate_temporal_facades()
