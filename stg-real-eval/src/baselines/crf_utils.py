import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_softmax,
        create_pairwise_gaussian,
        create_pairwise_bilateral,
    )
except Exception:  # pragma: no cover
    dcrf = None


def dense_crf_refine(img_rgb: np.ndarray, prob_fg: np.ndarray, iters: int = 5):
    """
    Refine a foreground probability map (H,W) with DenseCRF.
    Returns a binary mask uint8 {0,1}.
    """
    if dcrf is None:
        return (prob_fg > 0.5).astype(np.uint8)
    H, W = prob_fg.shape
    probs = np.stack([1.0 - prob_fg, prob_fg], axis=0).astype(np.float32)
    unary = unary_from_softmax(probs).reshape((2, -1))
    crf = dcrf.DenseCRF2D(W, H, 2)
    crf.setUnaryEnergy(unary)
    feats_g = create_pairwise_gaussian(sdims=(3, 3), shape=(H, W))
    crf.addPairwiseEnergy(feats_g, compat=3)
    feats_bi = create_pairwise_bilateral(
        sdims=(60, 60), schan=(10, 10, 10), img=img_rgb, chdim=2
    )
    crf.addPairwiseEnergy(feats_bi, compat=5)
    Q = crf.inference(iters)
    refined = np.array(Q).reshape((2, H, W))[1]
    return (refined > 0.5).astype(np.uint8)
