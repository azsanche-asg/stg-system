# STG Synthetic Evaluation

This repository evaluates the Spatio-Temporal Scene Grammar (ST-SG) on synthetic, fully controlled datasets
to measure structural accuracy (split/repeat detection, compactness) and temporal persistence.

### Objectives
1. Quantitatively validate grammar induction with known ground truth.
2. Test ablations: -noRepeat, -noSplit, -noTemporal.
3. Produce tables and figures for Table 1 in the CVPR paper.

### Folder layout
- `datasets/` – JSON/NPZ files with procedural facades/streets and GT grammars.
- `configs/` – YAML experiment configs.
- `src/` – evaluation code.
- `metrics/` – reusable metric definitions.
- `outputs/` – generated tables and plots.
- `plots/` – figures for the paper.

### Setup
```bash
conda create -n stg_eval python=3.10
conda activate stg_eval
pip install -r requirements.txt
Run
bash
Copy code
python src/run_eval.py --config configs/synthetic_facades.yaml
```
