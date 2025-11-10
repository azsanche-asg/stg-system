import numpy as np
import cv2
from PIL import Image


def _sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _estimate_vertical_repeats(gray):
    gx, _ = _sobel(gray)
    sig = np.mean(np.abs(gx), axis=0)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1 :]
    ac[:5] = 0
    k = np.argmax(ac)
    if k <= 0:
        return 1
    repeats = max(1, int(gray.shape[1] / max(8, k)))
    return int(np.clip(repeats, 1, 50))


def _estimate_horizontal_splits(gray):
    _, gy = _sobel(gray)
    sig = np.mean(np.abs(gy), axis=1)
    thr = sig.mean() + 0.75 * sig.std()
    peaks = np.where(sig > thr)[0]
    if len(peaks) < 2:
        return 1
    _sep = max(4, gray.shape[0] // 20)
    count = 1
    last = peaks[0]
    for p in peaks[1:]:
        if p - last > _sep:
            count += 1
            last = p
    return int(np.clip(count, 1, 30))


def _edge_mask(gray):
    e = cv2.Canny(gray, 100, 200)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
    return (e > 0).astype(np.uint8)


def infer_raster_baseline(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    floors = _estimate_horizontal_splits(gray)
    repeats = _estimate_vertical_repeats(gray)
    mask = _edge_mask(gray)

    grammar = {
        "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
        "repeats": [int(floors), int(repeats)],
        "depth": 2,
        "persist_ids": [],
        "motion": [],
        "proxy_mask": mask.astype(np.uint8).tolist(),
    }
    return grammar
