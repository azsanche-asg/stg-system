# STG Baselines

Lightweight baselines for evaluating Spatio-Temporal Scene Grammar (ST-SG)
on synthetic datasets.

Implements:
1. BSP-Depth  – splits image using depth gradients.
2. Seg-Repeat – detects repetition patterns along axes.
3. SceneGraph-Lite – builds a symbolic adjacency graph from detected regions.

Outputs JSONs compatible with `stg-synthetic-eval`.
