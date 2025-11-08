import numpy as np


def ade_fde(traj_pred, traj_gt):
    """Average/Final displacement error for simple 2D trajectories."""
    if len(traj_pred) == 0 or len(traj_gt) == 0 or len(traj_pred) != len(traj_gt):
        return np.nan, np.nan
    diffs = [np.linalg.norm(np.array(p) - np.array(g)) for p, g in zip(traj_pred, traj_gt)]
    ade = float(np.mean(diffs))
    fde = float(diffs[-1])
    return ade, fde


def replay_iou(mask_seq_pred, mask_seq_gt):
    """IoU across a sequence; both lists of HxW bool arrays."""
    if not mask_seq_pred or not mask_seq_gt or len(mask_seq_pred) != len(mask_seq_gt):
        return np.nan
    ious = []
    for p, g in zip(mask_seq_pred, mask_seq_gt):
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else np.nan


def edit_consistency_iou(before_pred, after_pred, edit_mask):
    """
    IoU between predicted structures before/after an edit, restricted to the edit region.
    All inputs are HxW bool arrays (or broadcastable).
    """
    region = edit_mask.astype(bool)
    bp = before_pred.astype(bool) & region
    ap = after_pred.astype(bool) & region
    inter = np.logical_and(bp, ap).sum()
    union = np.logical_or(bp, ap).sum()
    return float(inter / union) if union else np.nan

