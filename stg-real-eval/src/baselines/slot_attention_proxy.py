import numpy as np
import torch
import torchvision
from PIL import Image


class SlotAttentionWrapper(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/slot_attention_pytorch",
            "slot_attention_clevr",
            pretrained=True,
        )
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def infer_slots(self, pil_img: Image.Image, max_slots: int = 7):
        x = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(self.device)
        out = self.model(x)
        masks = out["masks"][0].cpu().numpy()
        num_slots = min(masks.shape[0], max_slots)
        ys, xs = np.where(masks.sum(0) > 0)
        if ys.size == 0 or xs.size == 0:
            floors = repeats = 1
        else:
            v_bins = np.unique(np.digitize(xs, np.linspace(0, masks.shape[2], num_slots)))
            h_bins = np.unique(np.digitize(ys, np.linspace(0, masks.shape[1], num_slots)))
            floors, repeats = len(h_bins), len(v_bins)
        mask_union = (masks.sum(0) > 0).astype(np.uint8)
        return {
            "rules": [f"Split_y_{floors}", f"Repeat_x_{repeats}"],
            "repeats": [int(floors), int(repeats)],
            "depth": 2,
            "persist_ids": [],
            "motion": [],
            "proxy_mask": mask_union.tolist(),
        }


def infer_slot_baseline(pil_img: Image.Image):
    global _SLOT_MODEL
    if "_SLOT_MODEL" not in globals():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _SLOT_MODEL = SlotAttentionWrapper(device=device)
    return _SLOT_MODEL.infer_slots(pil_img)
