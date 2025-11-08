"""
CMP Facade tiny loader (≤10 images).
Assumes {root} contains a flat set of facade JPG/PNG files.
"""
from pathlib import Path
from typing import List

from .common import Scene, to_frame


class CMPFacade:
    def __init__(self, root: str, max_images: int = 10):
        self.root = Path(root)
        self.max_images = max_images

    def list_scenes(self) -> List[Scene]:
        imgs = sorted(list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")))[: self.max_images]
        if not imgs:
            return []
        sid = "cmp_10"
        return [
            Scene(
                dataset="cmp-facade",
                scene_id=sid,
                frames=[to_frame(p, sid, p.stem) for p in imgs],
            )
        ]

