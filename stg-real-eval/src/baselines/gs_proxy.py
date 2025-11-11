import numpy as np
import cv2
from PIL import Image


class GSProxyWrapper:
    """Lightweight 3D Gaussian proxy baseline."""

    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def _depth_to_cloud(self, depth):
        H, W = depth.shape
        ys, xs = np.mgrid[0:H, 0:W]
        zs = depth / (depth.max() + 1e-6)
        pts = np.stack((xs / W, ys / H, zs), axis=-1).reshape(-1, 3)
        return pts

    def _fit_planes(self, pts, n_planes=3):
        if len(pts) < 50:
            return []
        planes = []
        pts_remain = pts.copy()
        for _ in range(n_planes):
            if len(pts_remain) < 50:
                break
            idx = np.random.choice(len(pts_remain), min(200, len(pts_remain)), replace=False)
            sample = pts_remain[idx].astype(np.float32)
            ok, normal = cv2.fitPlane(sample)
            if not ok:
                break
            d = np.abs(pts_remain @ normal)
            inliers = d < self.sigma
            planes.append(normal)
            pts_remain = pts_remain[~inliers]
        return planes

    def infer_geometry(self, pil_img: Image.Image, depth_np: np.ndarray):
        cloud = self._depth_to_cloud(depth_np)
        planes = self._fit_planes(cloud, n_planes=3)
        floors = len(planes) if planes else 1
        ys = cloud[:, 1]
        repeats = max(1, int(np.clip(len(np.unique(np.digitize(ys, np.linspace(0, 1, 6)))), 1, 10)))
        mask = (depth_np > np.median(depth_np)).astype(np.uint8)
        feats = cloud[: min(5000, len(cloud))]
        if len(feats) > 1:
            norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
            sim = norm @ norm.T
            avg_sim = float(np.tril(sim, -1).mean())
        else:
            avg_sim = float("nan")
        return {
            "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
            "repeats": [floors, repeats],
            "depth": 2,
            "proxy_mask": mask.tolist(),
            "cluster_feats": feats.tolist(),
            "avg_sim": avg_sim,
        }


def infer_gs_proxy(pil_img, depth_np):
    global _GS_MODEL
    if "_GS_MODEL" not in globals():
        _GS_MODEL = GSProxyWrapper()
    return _GS_MODEL.infer_geometry(pil_img, depth_np)
