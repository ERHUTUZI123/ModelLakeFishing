# PRODUCTION manifest (current default outputs)

**As of 2026-06-20.** These are the production-default artifacts. Everything else in
`legacy/` is ablation/history, not the default. All entrypoints default to these;
legacy graphs/checkpoints stay reproducible via `--pt` / `--ckpt`.

## Current production files
| role | file |
|---|---|
| Stage-1 graph | `stage1BuildTransferGraph/hgraph_diverse_xd0.pt` |
| raw HGraph viz | `stage1BuildTransferGraph/hgraph_diverse_xd0_viz.png` |
| Stage-2 checkpoint | `artifacts/stage2_diverse_xd0_candidate.pt` |
| family vocab (model side) | `artifacts/stage2_diverse_xd0_candidate.family_vocab.csv` |
| task_type vocab (dataset side) | `artifacts/stage2_diverse_xd0_candidate.task_type_vocab.csv` |
| trained-embedding viz | `artifacts/stage2_diverse_xd0_embedding_viz.png` |
| report | `artifacts/stage2_diverse_xd0_report.md` |
| dataset provenance | `stage1BuildTransferGraph/diverse_zoo_manifest.{csv,json}` |

## Guarantees (verified)
- **diverse zoo**: 306 models, 24 datasets (177 local + 127 HF-harvested + lineage hubs).
- **xm0 enabled**: `data["model"].x.shape == (306, 448)` (frozen `[e_name‖e_desc]`,
  real features, not smoke random).
- **xd0 enabled**: `data["dataset"].x.shape == (24, 2310)` (multi-view
  `[e_domain‖e_label‖e_card‖e_stats]`); `xd0_meta` present;
  `data["dataset"].{task_type_id,n_class_bucket_id,arity_id}` present.
- **DatasetNodeEncoder ON**: the trained checkpoint's `arch` carries `num_task_types`
  etc.; `model.use_dataset_encoder is True`.
- **no node-id embedding**: only semantic discrete tables (size/family on the model
  side; task_type/n_class/arity on the dataset side). Inductive contract intact.
- **checkpoint roundtrip OK**: reload reproduces z exactly; family_vocab +
  task_type_vocab bindings validated (orphan-row guard).
- `check_stage2_contract.py --pt hgraph_diverse_xd0.pt` → CONTRACT OK.

## Default entrypoints (now point here)
- `train_diverse.py`        → `--pt hgraph_diverse_xd0.pt`, `--out stage2_diverse_xd0_candidate.pt`
- `visualize_trained.py`    → `--pt hgraph_diverse_xd0.pt`, `--ckpt stage2_diverse_xd0_candidate.pt`,
                              `--out stage2_diverse_xd0_embedding_viz.png`

## NOT switched (legacy by design)
`experiment.py` (zoo robustness matrix), `train.py` (mechanism smoke), and
`model.load_hgraph()`'s default still target the **zoo** graph + `assert num_families==136`.
They are legacy ablation/smoke harnesses, not the production trainer. Production
training/visualization goes through the two entrypoints above.

## Headline result (vs the xm0-only diverse baseline)
x_d cos-gram effective rank 3.34 → **7.11**; mean pairwise cosine 0.39 → **0.15**;
tau_macro 0.213 → **0.223** (std halved). BUT z_d participation flat (1.32→1.38) and
z_m participation down (2.04→1.45): the participation bottleneck moved off the dataset
representation onto the Stage-2 objective + dataset count. Full honesty in
`stage2_diverse_xd0_report.md`.
