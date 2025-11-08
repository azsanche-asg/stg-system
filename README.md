# STG-System (Monorepo)

This repository unifies all components:
- **stg-procedural-data** – synthetic & temporal data generation  
- **stg-baselines** – baseline models for comparison  
- **stg-stsg-model** – ST-SG model, ablation suite, and inference  
- **stg-synthetic-eval** – evaluation, metrics, visualization, logging  
- **stg-real-eval** – Block B real “mini” evaluation (nuScenes-mini, Cityscapes-Seq, CMP)  
  ```bash
  python stg-real-eval/src/run_eval_real.py --config stg-real-eval/configs/block_b_cityscapes.yaml
  ```

### Structure
stg-system/
├── stg-procedural-data/
├── stg-baselines/
├── stg-stsg-model/
├── stg-synthetic-eval/
├── stg-real-eval/
└── scripts/

pgsql
Copy code

### Snapshot
Frozen as **v1_static_baseline** (Block A complete).
