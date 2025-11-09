import numpy as np

from .temporal_tracker import FlowTracker


def ade_fde(traj_pred, traj_gt):
    """Average/Final displacement error for simple 2D trajectories."""
    if len(traj_pred) == 0 or len(traj_gt) == 0 or len(traj_pred) != len(traj_gt):
        return np.nan, np.nan
    diffs = [np.linalg.norm(np.array(p) - np.array(g)) for p, g in zip(traj_pred, traj_gt)]
    ade = float(np.mean(diffs))
    fde = float(diffs[-1])
    return ade, fde


def ade_fde_from_flow(frames):
    """
    Compute ADE/FDE from optical-flow displacement magnitudes.
    frames: list of RGB np.ndarray images.
    """
    if not frames:
        return np.nan, np.nan
    tracker = FlowTracker()
    disps = []
    for frame in frames:
        flow = tracker.update(frame)
        if flow is not None:
            mag = np.linalg.norm(flow, axis=-1)
            disps.append(mag.mean())
    if not disps:
        return np.nan, np.nan
    ade = float(np.mean(disps))
    fde = float(disps[-1])
    return ade, fde


def replay_iou(mask_seq_pred):
    """
    Temporal IoU of a predicted mask sequence warped with flow.
    """
    if not mask_seq_pred or len(mask_seq_pred) < 2:
        return np.nan
    tracker = FlowTracker()
    warped_ious = []
    prev_frame = None
    for idx, mask in enumerate(mask_seq_pred):
        mask_bool = mask.astype(bool)
        rgb_frame = np.repeat(mask_bool[..., None], 3, axis=-1).astype(np.uint8) * 255
        flow = tracker.update(rgb_frame)
        if flow is None or prev_frame is None:
            prev_frame = mask_bool
            continue
        warped = tracker.warp_mask(prev_frame.astype(np.uint8), flow) > 0.5
        inter = np.logical_and(warped, mask_bool).sum()
        union = np.logical_or(warped, mask_bool).sum()
        if union > 0:
            warped_ious.append(inter / union)
        prev_frame = mask_bool
    return float(np.mean(warped_ious)) if warped_ious else np.nan


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
