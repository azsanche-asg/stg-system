from pathlib import Path


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root() -> Path:
    return ensure_dir(Path("cache") / "block_b")


def dataset_cache_root(dataset_name: str) -> Path:
    return ensure_dir(cache_root() / dataset_name)

