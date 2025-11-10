import numpy as np


def _cos(a, b):
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    return float(np.dot(a, b) / (na * nb))


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def match_slots_across_frames(slot_masks_seq, slot_embs_seq, alpha=0.5):
    """
    Greedy matching frame-to-frame, returning per-frame union masks.
    """
    if not slot_masks_seq:
        return []
    per_frame_union = [np.logical_or.reduce(slot_masks_seq[0], axis=0)]
    prev_masks = slot_masks_seq[0]
    prev_embs = slot_embs_seq[0]
    for t in range(1, len(slot_masks_seq)):
        cur_masks = slot_masks_seq[t]
        cur_embs = slot_embs_seq[t]
        used = set()
        matched = []
        for i, pm in enumerate(prev_masks):
            best, bj = -1.0, -1
            for j, cm in enumerate(cur_masks):
                if j in used:
                    continue
                score = alpha * _iou(pm, cm) + (1 - alpha) * _cos(prev_embs[i], cur_embs[j])
                if score > best:
                    best, bj = score, j
            if bj >= 0:
                used.add(bj)
                matched.append(cur_masks[bj])
        if matched:
            per_frame_union.append(np.logical_or.reduce(np.stack(matched, axis=0), axis=0))
        else:
            per_frame_union.append(np.logical_or.reduce(cur_masks, axis=0))
        prev_masks = cur_masks
        prev_embs = cur_embs
    return per_frame_union
