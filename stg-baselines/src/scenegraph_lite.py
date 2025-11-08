import json
import os
import numpy as np
from tqdm import tqdm
from src.utils.naming_utils import make_pred_filename


def run_scenegraph(mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in tqdm(sorted(os.listdir(mask_dir)), desc='SceneGraph-Lite'):
        if not fname.endswith('_mask.npy'):
            continue
        mask_path = os.path.join(mask_dir, fname)
        mask = np.load(mask_path)
        objs = np.unique(mask)[1:]
        rules = []
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                ya, xa = np.mean(np.argwhere(mask == a), 0)
                yb, xb = np.mean(np.argwhere(mask == b), 0)
                if xa < xb:
                    rules.append(f"{a}_leftof_{b}")
                if ya < yb:
                    rules.append(f"{a}_above_{b}")
        pred = {
            "rules": rules,
            "repeats": [],
            "depth": 1,
            "persist_ids": [],
            "motion": []
        }
        out_path = os.path.join(out_dir, make_pred_filename(fname))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(pred, f)
