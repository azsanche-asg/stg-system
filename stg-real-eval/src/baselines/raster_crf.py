import numpy as np
from PIL import Image

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    _HAVE_CRF = True
except Exception:  # pragma: no cover
    _HAVE_CRF = False


def _softmax_logits(logits, axis=0):
    x = logits - logits.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-8)


def apply_densecrf(
    rgb: np.ndarray,
    logits: np.ndarray,
    iters: int = 5,
    sxy_gauss=3,
    compat_gauss=3,
    sxy_bilateral=60,
    compat_bilateral=10,
    srgb=5,
):
    """Return refined probability maps (C,H,W)."""
    H, W = rgb.shape[:2]
    if logits.ndim == 2:
        logits = np.stack([1 - logits, logits], axis=0)
    probs = _softmax_logits(logits, axis=0)

    if not _HAVE_CRF:
        return probs

    C = probs.shape[0]
    d = dcrf.DenseCRF2D(W, H, C)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral, srgb=srgb, rgbim=rgb, compat=compat_bilateral
    )
    Q = d.inference(iters)
    return np.array(Q).reshape(C, H, W)


def raster_with_crf(rgb_im: Image.Image):
    """Edge unary + DenseCRF refinement."""
    import cv2

    im = np.array(rgb_im.convert("RGB"))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)

    logits = np.stack([1 - mag, mag], axis=0)
    probs = apply_densecrf(im.astype(np.uint8), logits, iters=5)
    mask = (probs[1] > probs[0]).astype(np.uint8)

    sig = np.mean(np.abs(gx), axis=0)
    peaks = int((sig > (sig.mean() + sig.std())).sum())

    return {
        "rules": ["Split_y_1", f"Repeat_x_{max(1, peaks)}"],
        "repeats": [1, int(max(1, peaks))],
        "depth": 2,
        "persist_ids": [],
        "motion": [],
        "proxy_mask": mask.tolist(),
        "repeat_x_peaks": peaks,
    }
