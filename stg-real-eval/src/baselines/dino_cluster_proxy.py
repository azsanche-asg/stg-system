import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.cluster import KMeans


class DINOClusterWrapper(torch.nn.Module):
    """DINO v2 feature clustering baseline."""

    def __init__(self, device="cpu", num_clusters=6):
        super().__init__()
        print("🧠  Loading DINO v2 ViT-S/14 backbone from Torch Hub...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        self.device = device
        self.num_clusters = num_clusters

    @torch.no_grad()
    def infer_regions(self, pil_img: Image.Image):
        """Cluster DINO tokens into region masks without assuming square grids."""
        import math

        img = torchvision.transforms.functional.resize(pil_img, (224, 224))
        x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(self.device)

        feats = self.model.get_intermediate_layers(x, n=1)[0]
        tokens = feats[0, 1:, :].cpu().numpy()
        N, D = tokens.shape

        H = int(math.floor(math.sqrt(N)))
        W = int(math.ceil(N / H))
        if H * W != N:
            pad = np.zeros((H * W - N, D), dtype=tokens.dtype)
            tokens = np.concatenate([tokens, pad], axis=0)

        km = KMeans(self.num_clusters, n_init=3, random_state=0).fit(tokens)
        labels = km.labels_.reshape(H, W)
        masks = [(labels == i).astype(np.uint8) for i in range(self.num_clusters)]
        union = (labels >= 0).astype(np.uint8)

        ys, xs = np.where(union > 0)
        v_bins = np.unique(np.digitize(xs, np.linspace(0, W, self.num_clusters)))
        h_bins = np.unique(np.digitize(ys, np.linspace(0, H, self.num_clusters)))
        floors, repeats = len(h_bins), len(v_bins)

        return {
            "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
            "repeats": [int(floors), int(repeats)],
            "depth": 2,
            "persist_ids": [],
            "motion": [],
            "proxy_mask": union.tolist(),
            "slot_masks": [m.tolist() for m in masks],
        }


def infer_dino_cluster(pil_img: Image.Image):
    global _DINO_MODEL
    if "_DINO_MODEL" not in globals():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DINO_MODEL = DINOClusterWrapper(device=device)
    return _DINO_MODEL.infer_regions(pil_img)
