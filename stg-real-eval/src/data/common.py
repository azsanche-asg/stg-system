from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class Frame:
    image_path: Path
    scene_id: str
    frame_id: str
    # Optional metadata (pose, boxes, etc.)
    meta: Dict[str, Any]


@dataclass
class Scene:
    dataset: str
    scene_id: str
    frames: List[Frame]


# Tiny helper to keep datasets consistent
def to_frame(image_path: Path, scene_id: str, frame_id: str, meta=None) -> Frame:
    return Frame(image_path=image_path, scene_id=scene_id, frame_id=frame_id, meta=meta or {})

