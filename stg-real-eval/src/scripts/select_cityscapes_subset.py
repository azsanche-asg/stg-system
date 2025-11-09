"""
Cityscapes subset selector for Block B evaluation.
Allows controlled, reproducible selection of a small subset of frames/cities
from either static (leftImg8bit), sequence (leftImg8bit_sequence), or demoVideo splits.

Usage (examples):
-----------------
# Select 2 towns, 20 frames total, store persistently in repo
python stg-real-eval/src/scripts/select_cityscapes_subset.py \
  --root ~/data/cityscapes/leftImg8bit_sequence/train \
  --num_towns 2 --num_frames 20 \
  --dest ./stg-real-eval/datasets/cityscapes_subset

# Same but store temporarily (Colab / ephemeral)
python stg-real-eval/src/scripts/select_cityscapes_subset.py \
  --root /content/leftImg8bit_sequence/demoVideo \
  --num_towns 1 --num_frames 10 \
  --dest /tmp/cityscapes_subset --temporary

Notes:
------
- Supports both "train" and "demoVideo" splits.
- Keeps directory structure: dest/<town>/<frame>.png
- Writes manifest.json summarizing the selection.
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def select_images(root: Path, num_towns: int, num_frames: int) -> Dict[str, List[str]]:
    """Randomly pick towns and frames; returns {town: [paths]}"""
    towns = sorted([p for p in root.glob("*") if p.is_dir()])
    if not towns:
        raise FileNotFoundError(f"No towns found under {root}")
    chosen_towns = random.sample(towns, k=min(num_towns, len(towns)))
    selection = {}
    for t in chosen_towns:
        imgs_all = list(t.glob("*.png")) + list(t.glob("*.jpg"))
        imgs = sorted(
            [p for p in imgs_all if p.is_file() and not p.name.startswith(".") and not p.name.startswith("._")]
        )
        if not imgs:
            continue
        selected_imgs = random.sample(imgs, k=min(num_frames, len(imgs)))
        selection[t.name] = [str(p) for p in selected_imgs]
    return selection


def copy_selection(selection: Dict[str, List[str]], dest: Path):
    for town, paths in selection.items():
        town_dest = dest / town
        town_dest.mkdir(parents=True, exist_ok=True)
        for src in paths:
            shutil.copy(src, town_dest / Path(src).name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Root of leftImg8bit or leftImg8bit_sequence (train or demoVideo).",
    )
    ap.add_argument("--num_towns", type=int, default=1)
    ap.add_argument("--num_frames", type=int, default=20)
    ap.add_argument("--dest", required=True, help="Destination folder for subset.")
    ap.add_argument(
        "--temporary",
        action="store_true",
        help="If set, marks data as temporary (skips repo commit instructions).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root).expanduser()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    selection = select_images(root, args.num_towns, args.num_frames)
    copy_selection(selection, dest)

    manifest = {
        "root": str(root),
        "dest": str(dest),
        "num_towns": args.num_towns,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "selection": selection,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\n✅ Subset created with {sum(len(v) for v in selection.values())} images "
        f"from {len(selection)} towns."
    )
    print(f"Manifest saved to: {dest/'manifest.json'}")

    if not args.temporary:
        print("\n👉 To persist this subset in the repo, commit the folder:")
        print(f"   git add {dest}")
        print(
            f"   git commit -m 'Add Cityscapes subset ({args.num_towns} towns, {args.num_frames} frames)'"
        )


if __name__ == "__main__":
    main()
