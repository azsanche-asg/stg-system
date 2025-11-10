import numpy as np
import torch
import torchvision
from PIL import Image


def _slot_color_embed(img_rgb, mask):
    m = mask > 0
    if m.sum() == 0:
        return np.zeros(3, dtype=np.float32)
    col = img_rgb[m].mean(axis=0)
    return col.astype(np.float32)


class SlotAttentionWrapper(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = (
            torch.hub.load(
                "facebookresearch/slot_attention_pytorch",
                "slot_attention_clevr",
                pretrained=True,
            )
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def infer_slots(self, pil_img: Image.Image, max_slots: int = 7):
        x = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(self.device)
        out = self.model(x)
        masks = out["masks"][0].cpu().numpy()
        K, H, W = masks.shape
        K = min(K, max_slots)
        masks = masks[:K] > 0.5
        ys, xs = np.where(masks.sum(0) > 0)
        if xs.size == 0:
            floors = repeats = 1
        else:
            v_bins = np.unique(np.digitize(xs, np.linspace(0, W, max_slots)))
            h_bins = np.unique(np.digitize(ys, np.linspace(0, H, max_slots)))
            floors, repeats = len(h_bins), len(v_bins)
        img_rgb = np.array(pil_img.convert("RGB"))
        slot_embs = [_slot_color_embed(img_rgb, m) for m in masks]
        mask_union = (masks.sum(0) > 0).astype(np.uint8)
        return {
            "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
            "repeats": [int(floors), int(repeats)],
            "depth": 2,
            "persist_ids": [],
            "motion": [],
            "proxy_mask": mask_union.tolist(),
            "slot_masks": [m.astype(np.uint8).tolist() for m in masks],
            "slot_embs": [e.tolist() for e in slot_embs],
        }


def infer_slot_baseline(pil_img: Image.Image):
    global _SLOT_MODEL
    if "_SLOT_MODEL" not in globals():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SLOT_MODEL = SlotAttentionWrapper(device=device)
    return _SLOT_MODEL.infer_slots(pil_img)
