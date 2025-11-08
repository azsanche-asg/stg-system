import os
import re

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / 'stg-stsg-model' / 'src'))


def stem(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"(\.(png|jpg|jpeg|npy))$", "", base)
    base = base.replace("_depth", "").replace("_mask", "")
    return base
