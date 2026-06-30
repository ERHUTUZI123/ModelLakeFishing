# ModelLakeFishing — scalable model recommendation for million-scale model lakes

## Environment
```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```
(On this machine always use the venv python directly:
`./ModelLakeFishing/.venv/Scripts/python.exe ...`)

---

## PRODUCTION pipeline (current default: diverse zoo + xm0 + xd0)

The production stack uses the **diverse zoo** with **real xm0 model features** AND
**xd0 multi-view dataset features** (DatasetNodeEncoder ON). See
`stage2TrainGraphSAGE/artifacts/PRODUCTION.md` for the authoritative manifest.

```
# --- Stage 1: build the production graph (diverse zoo, xm0 + xd0) ---
cd stage1BuildTransferGraph
# (re)harvest the diverse zoo into dataset_embed/data_diverse/ (provenance: diverse_zoo_manifest.*)
python build_diverse_zoo.py                       # or --offline to reuse hf_cache/
# build the graph from the isolated diverse data dir
MLF_DATA_DIR=$PWD/dataset_embed/data_diverse \
  python build_graph.py --contain_model_feature True --contain_rich_dataset_feature True \
                        --out hgraph_diverse_xd0.pt
python check_stage2_contract.py --pt hgraph_diverse_xd0.pt
python visualize_hgraph.py --pt hgraph_diverse_xd0.pt --out hgraph_diverse_xd0_viz.png

# --- Stage 2: train the candidate (defaults already point at the xd0 graph) ---
cd ..
python -m ModelLakeFishing.stage2TrainGraphSAGE.train_diverse --seeds 3 --epochs 25
#   -> artifacts/stage2_diverse_xd0_candidate.pt (+ .family_vocab.csv, .task_type_vocab.csv)

# --- trained-embedding visualization (defaults already point at the xd0 ckpt) ---
python -m ModelLakeFishing.stage2TrainGraphSAGE.visualize_trained
#   -> artifacts/stage2_diverse_xd0_embedding_viz.png

# --- diagnostics ---
python -m ModelLakeFishing.stage2TrainGraphSAGE.diverse_diagnostics \
  --pt stage1BuildTransferGraph/hgraph_diverse_xd0.pt \
  --ckpt stage2TrainGraphSAGE/artifacts/stage2_diverse_xd0_candidate.pt
```

Production report: `stage2TrainGraphSAGE/artifacts/stage2_diverse_xd0_report.md`.

---

## LEGACY / baselines (kept for reproducibility — not the default)

Archived outputs live under `stage2TrainGraphSAGE/artifacts/legacy/` and
`stage1BuildTransferGraph/legacy/` (see their README.md). Every entrypoint still
accepts `--pt` / `--ckpt` so legacy baselines remain reproducible:

```
cd stage1BuildTransferGraph
# old zoo (non-diverse), real xm0:
python build_graph.py --contain_model_feature True  --out hgraph_zoo_xm0.pt
python visualize_hgraph.py --pt hgraph_zoo_xm0.pt --out hgraph_zoo_xm0_viz.png
# smoke / no-xm0 (random model features):
python build_graph.py --contain_model_feature False --out hgraph_zoo.pt

# diverse zoo WITHOUT xd0 (xm0-only baseline):
MLF_DATA_DIR=$PWD/dataset_embed/data_diverse \
  python build_graph.py --contain_model_feature True --out hgraph_diverse_xm0.pt

# train on a legacy graph (explicit paths):
cd ..
python -m ModelLakeFishing.stage2TrainGraphSAGE.train_diverse \
  --pt stage1BuildTransferGraph/hgraph_diverse_xm0.pt \
  --out stage2TrainGraphSAGE/artifacts/legacy/stage2_diverse_candidate.pt
```

`experiment.py` (zoo robustness matrix) and `train.py` (mechanism smoke) remain
pinned to the zoo graph **by design** — they are legacy ablation/smoke harnesses,
not the production trainer (which is `train_diverse.py`).
