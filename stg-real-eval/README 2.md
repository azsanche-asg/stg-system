# stg-real-eval (Block B)
Real-data "mini" evaluation for the STG System.

**Datasets (tiny):**
- nuScenes-mini (10 short scenes)
- Cityscapes-Seq (+ Cityscapes single frames)
- CMP Facade (≤10 images)

**Pipeline:**
1) `scripts/extract_features.py` → cache MiDaS/SAM/DINO/CLIP at 512².
2) `run_eval_real.py --config configs/block_b_*.yaml`
3) `scripts/make_summary_tables.py` → Table 2/3 JSON (means ± CI).
4) `scripts/make_overlay.py` → qualitative panels.

This module mirrors `stg-synthetic-eval` but reads real data and adds temporal metrics.

