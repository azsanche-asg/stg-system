import numpy as np
import torch
import torchvision
from PIL import Image

from .slot_attention_loader import SlotAttentionWrapper


def _slot_color_embed(img_rgb, mask):
    m = mask > 0
    if m.sum() == 0:
        return np.zeros(3, dtype=np.float32)
    return img_rgb[m].mean(axis=0).astype(np.float32)


class SlotAttentionProxy(torch.nn.Module):
    def __init__(self, checkpoint="slot_attention_clevr", device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.wrapper = SlotAttentionWrapper(checkpoint_name=checkpoint, device=device)

    @torch.no_grad()
    def infer(self, pil_img: Image.Image, max_slots: int = 7):
        x = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(self.device)
        outputs = self.wrapper(x)
        masks = outputs["masks"][0].cpu().numpy()
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
    global _SLOT_PROXY
    if "_SLOT_PROXY" not in globals():
        _SLOT_PROXY = SlotAttentionProxy()
    return _SLOT_PROXY.infer(pil_img)
