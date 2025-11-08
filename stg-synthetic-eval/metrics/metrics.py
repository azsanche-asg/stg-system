import numpy as np


def rule_f1(pred, gt):
    tp = len(set(pred["rules"]) & set(gt["rules"]))
    fp = len(set(pred["rules"]) - set(gt["rules"]))
    fn = len(set(gt["rules"]) - set(pred["rules"]))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def regularity_error(pred, gt):
    return np.mean(np.abs(np.array(pred["repeats"]) - np.array(gt["repeats"])))


def mdl_score(pred):
    return len(pred["rules"]) + 0.5 * pred["depth"]


def persistence_score(pred, gt):
    if 'persist_ids' not in gt or not gt['persist_ids']:
        return np.nan
    pred_ids = set(pred.get('persist_ids', []))
    gt_ids = set(gt['persist_ids'])
    if not gt_ids:
        return np.nan
    return len(pred_ids & gt_ids) / len(gt_ids)


def motion_error(pred, gt):
    """Compute average motion vector error; tolerant of missing or list fields."""
    gt_motion = gt.get('motion', {})
    pr_motion = pred.get('motion', {})
    if not isinstance(gt_motion, dict) or not isinstance(pr_motion, dict):
        return np.nan
    diffs = []
    for key, gt_vec in gt_motion.items():
        pr_vec = pr_motion.get(key) if isinstance(pr_motion, dict) else None
        if pr_vec is not None:
            try:
                diffs.append(np.linalg.norm(np.array(pr_vec) - np.array(gt_vec)))
            except Exception:
                continue
    return np.mean(diffs) if diffs else np.nan
