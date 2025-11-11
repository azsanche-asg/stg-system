import torch
from object_discovery.models.slot_attention import SlotAttentionAutoEncoder
from object_discovery.utils import load_model_from_hub


class SlotAttentionWrapper(torch.nn.Module):
    """Wrap HHousen pretrained Slot Attention model."""

    def __init__(self, checkpoint_name="slot_attention_clevr", device="cpu"):
        super().__init__()
        self.device = device
        print(
            f"[SlotAttentionWrapper] Loading pretrained model '{checkpoint_name}' from HHousen/object-discovery-pytorch ..."
        )
        try:
            self.model = load_model_from_hub(checkpoint_name, device=device)
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Auto-download failed ({exc}); falling back to random-init model.")
            self.model = SlotAttentionAutoEncoder().to(device)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
