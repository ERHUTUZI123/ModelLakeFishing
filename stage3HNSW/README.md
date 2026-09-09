# Current Stage 3 helper

This directory retains only `build_prior_sidecar.py`, the split-safe prior-sidecar builder used by the current 3M evaluation path. Current HNSW construction, calibration, and top-1,000 reranking live in `scale/export_ours.py`, `scale1m/eval_y2.py`, and `scale1m/eval_y4.py`.

The earlier small-lake serving stack and its artifacts are archived under [`legacy/pre_3m/stage3HNSW`](../legacy/pre_3m/stage3HNSW/).
