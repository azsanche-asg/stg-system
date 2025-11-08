import json
import os

import cv2
import numpy as np


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def save_depth(depth, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, depth.astype(np.float32))


def save_mask(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, mask.astype(np.int32))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
