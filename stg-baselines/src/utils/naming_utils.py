import os


def make_pred_filename(input_name: str, ext: str = ".json") -> str:
    """Return a clean prediction filename like scene_015_pred.json."""
    name = os.path.basename(input_name)
    # Remove known suffixes before extension
    for suf in ("_depth", "_mask"):
        if suf in name:
            name = name.replace(suf, "")
    name = os.path.splitext(name)[0]
    return f"{name}_pred{ext}"
