"""
Cityscapes-Seq tiny loader.
Assumes structure:
  {root}/leftImg8bit_sequence/train/<city>/*_leftImg8bit.png
For still frames, pass {root}/leftImg8bit/val/<city>/*.png
"""
from pathlib import Path
from typing import List

from .common import Scene, to_frame


class CityscapesSeq:
    def __init__(self, seq_root: str, still_root: str = None, take_every: int = 3, max_frames: int = 60):
        self.seq_root = Path(seq_root)
        self.still_root = Path(still_root) if still_root else None
        self.take_every = max(1, take_every)
        self.max_frames = max_frames

    def list_scenes(self) -> List[Scene]:
        scenes = []
        root = self.seq_root
        if not root.exists():
            return scenes

        # Case A: two-level layout, e.g. .../leftImg8bit_sequence/train/<city>/<frames>
        level2 = [p for p in root.glob("*/*") if p.is_dir()]
        # Case B: one-level layout, e.g. .../cityscapes_subset/<town>/<frames>
        level1 = [p for p in root.glob("*") if p.is_dir()] if not level2 else []

        seq_dirs = level2 if level2 else level1
        for seq_dir in sorted(seq_dirs):
            imgs = sorted(seq_dir.glob("*.png"))
            if not imgs:
                continue
            subs = imgs[::self.take_every][: self.max_frames]
            sid = seq_dir.name
            frames = [to_frame(p, sid, p.stem) for p in subs]
            scenes.append(Scene(dataset="cityscapes-seq", scene_id=sid, frames=frames))
        return scenes

    def list_stills(self) -> List[Scene]:
        if not self.still_root or not self.still_root.exists():
            return []
        city = next(iter(self.still_root.glob("*/*")), None)
        if not city:
            return []
        imgs = sorted(city.glob("*.png"))[:20]
        sid = f"cityscapes20_{city.name}"
        return [
            Scene(
                dataset="cityscapes",
                scene_id=sid,
                frames=[to_frame(p, sid, p.stem) for p in imgs],
            )
        ]
