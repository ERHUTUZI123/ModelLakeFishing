"""scale1m -- the R2/R3/R4 (100K / 500K / 1M) rung pipeline.

Runbook: docs/1M/100kplan.md   Ladder plan: docs/1M/plan.md

Module order (each stage's output is the next one's only input):
    hf_crawl.py          T2  crawl HF model metadata, downloads-desc, appendable
    verify_raw.py        T2  entry guard for T3: re-verify the frozen shards
    hf_canonicalize.py   T3  -> (id, size_b, family, lineage_base, layer)
    build_ladder.py      T3  CORE (frozen prefix) + HALO -> rung id table
    embed_lake.py        T4  CORE copied byte-for-byte, HALO embedded fresh
    build_graph_rung.py  T5  hgraph_<rung>.pt
    verify_rung_graph.py T5  iron-rule 1 assertions
    train_rung.py        T6
    export_rung.py       T7
    eval_rung.py         T8
"""
