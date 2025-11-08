import os


def footprint(model_json: str, runtime_s: float, gpu_mb: int = None):
    """Return a small dict used in Table 3."""
    try:
        size_bytes = os.path.getsize(model_json) if os.path.exists(model_json) else 0
    except Exception:
        size_bytes = 0
    return dict(model_size_bytes=size_bytes, runtime_s=float(runtime_s), gpu_mb=gpu_mb)

