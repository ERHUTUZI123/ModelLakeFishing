"""Current 3M-scale ModelLakeFishing pipeline.

Supported flow:
    crawl/canonicalize -> build_ladder_rf -> embed_lake_rf -> build_graph_rf
    -> train_rung -> export_rf -> eval_x6/eval_y2/eval_y4

The evaluated serving configuration is dense HNSW top-1,000 followed by the
split-safe deterministic task-evidence reranker in :mod:`eval_y2`. Superseded
rung builders and experimental evaluators are archived under ``legacy/``.
"""
