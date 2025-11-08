# STG Procedural Data Generator

This repository produces simple procedural urban scenes (facades and streets)
for evaluating Spatio-Temporal Scene Grammar (ST-SG).

Each generated sample includes:
- A synthetic RGB image.
- Depth map (normalized float array).
- Segmentation mask (label indices).
- Ground-truth grammar JSON (hierarchical Split/Repeat tree).
- Optional motion trajectories for temporal tests.

### Example usage
```bash
python src/generate_facades.py --n 50 --out outputs/facades
python src/generate_streets.py --n 20 --out outputs/streets
```
Outputs are fully compatible with stg-synthetic-eval and stg-baselines.
