# Legacy code archive

This directory contains historical implementations that are no longer part of the supported 3M retrieval path. Files were moved rather than deleted so their provenance and old experiment records remain inspectable.

| Directory | Archived scope |
|---|---|
| `100k_rung/` | Superseded 100K/500K/1M ladder construction, rung export, validation, and cluster orchestration |
| `pre_3m/` | Pre-3M Stage 1/2 graph experiments and the earlier small-lake Stage 3 serving stack |
| `post_y4_fusion/` | Z1/Z2 fusion experiments performed after the Y4 evidence cutoff |
| `evidence_exports/` | Superseded rendered evidence reports retained only for provenance |

Archive modules are not maintained as import-compatible packages and may still contain their original absolute imports and data paths. Use the contemporaneous documents under `docs/` to interpret them. Do not add a dependency from current code to this directory; if a historical helper becomes necessary, extract and test the minimal helper in a current module.

The current code map and supported commands are in the repository [`README.md`](../README.md).
