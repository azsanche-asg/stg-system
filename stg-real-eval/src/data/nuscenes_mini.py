"""
Tiny, path-based loader for nuScenes-mini.
Expects a folder layout like:
  {root}/samples/CAM_FRONT/*.jpg
  {root}/samples/CAM_FRONT_LEFT/*.jpg  (optional)
Poses/boxes are optional; if available, add to meta.
"""
from pathlib import Path
from typing import List

from .common import Scene, to_frame


class NuScenesMini:
    def __init__(self, root: str, take_every: int = 5, max_scenes: int = 10):
        self.root = Path(root)
        self.take_every = max(1, take_every)
        self.max_scenes = max_scenes

    def list_scenes(self) -> List[Scene]:
        cam = self.root / "samples" / "CAM_FRONT"
        if not cam.exists():
            return []
        # Group by prefix up to scene id (quick heuristic: parent is the scene)
        jpgs = sorted(cam.glob("*.jpg"))
        if not jpgs:
            return []
        # Make a single "mini" scene by sub-sampling
        subs = jpgs[::self.take_every]
        scene_id = "nuscenes_mini_front"
        frames = [to_frame(p, scene_id, p.stem) for p in subs]
        return [Scene(dataset="nuscenes-mini", scene_id=scene_id, frames=frames)][: self.max_scenes]

