import cv2
import numpy as np


class FlowTracker:
    """
    Lightweight temporal tracker using Farneback optical flow.
    Tracks points or masks across consecutive frames.
    """

    def __init__(self):
        self.prev_gray = None

    def update(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self.prev_gray = gray
        return flow

    def warp_mask(self, mask, flow):
        """Warp a binary mask using the computed flow field."""
        h, w = mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        dest_x = (grid_x + flow[..., 0]).astype(np.float32)
        dest_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            mask.astype(np.float32),
            dest_x,
            dest_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped
