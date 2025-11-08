import numpy as np


def delta_similarity(feat_matrix):
    """
    ΔSim = mean intra-cluster similarity – mean inter-cluster similarity.
    feat_matrix: (N, D) array of per-region feature embeddings.
    Returns float (higher is better, bounded [-1,1]).
    """
    if feat_matrix is None or len(feat_matrix) < 2:
        return np.nan
    sims = np.corrcoef(feat_matrix)
    intra = np.mean(np.diag(sims))
    N = len(feat_matrix)
    inter = (np.sum(sims) - np.trace(sims)) / (N * (N - 1))
    return float(intra - inter)


def purity(pred_labels, gt_labels):
    """
    Cluster purity = sum(max overlap per predicted cluster) / total samples.
    pred_labels, gt_labels: 1D integer arrays of same length.
    """
    if len(pred_labels) == 0 or len(gt_labels) == 0:
        return np.nan
    classes = np.unique(pred_labels)
    total = len(pred_labels)
    correct = 0
    for c in classes:
        mask = pred_labels == c
        if np.any(mask):
            most_common = np.bincount(gt_labels[mask]).argmax()
            correct += np.sum(gt_labels[mask] == most_common)
    return float(correct / total)


def facade_grid_score(mask, grid_shape=(3, 3)):
    """
    Simple façade-grid regularity measure:
    splits image into grid_shape cells, computes per-cell occupancy variance.
    Lower variance → higher regularity score (1 − normalized variance).
    """
    if mask is None or mask.size == 0:
        return np.nan
    H, W = mask.shape
    gh, gw = grid_shape
    h_step, w_step = max(1, H // gh), max(1, W // gw)
    occ = []
    for i in range(gh):
        for j in range(gw):
            cell = mask[i * h_step : (i + 1) * h_step, j * w_step : (j + 1) * w_step]
            if cell.size == 0:
                continue
            occ.append(cell.mean())
    if not occ:
        return np.nan
    occ = np.array(occ)
    var = np.var(occ)
    return float(1 - var / (occ.mean() ** 2 + 1e-8))
