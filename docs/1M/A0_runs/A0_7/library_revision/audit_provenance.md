# Independent provenance and bilingual structure audit

Audit scope: active `EVIDENCE_SOURCE_LIBRARY_en.md` and `_zh.md`, revised using A0.1–A0.7 evidence only. This audit changes neither active libraries nor filesystem aliases.

## 1. Preserve the historical authority before publishing the revision

The pre-revision English file has SHA-256 `6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd` and 51,538 bytes. The A0.1 frozen copy has the identical digest. The pre-revision Chinese file has SHA-256 `a148717b213bc99b013eddd8f2b0b6a290516b51397c513305768dcabab80f83`.

The A0.6 Linux protocol references the English authority at `/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing/docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md`. The Windows compatibility junction originally points that formal checkout to the active repository. Updating active English bytes while retaining that junction would break `scale1m.recompute_a0.verify_manifest`: it explicitly hashes the protocol authority and recursively hashes all manifest file records.

The A0.6 raw manifest contains **14 bound files beneath the formal-checkout prefix**: the historical English authority plus the following 13 code files:

- `scale1m/eval_y2.py`
- `scale1m/a0_evaluation.py`
- `scale1m/eval_rf.py`
- `scale1m/baselines.py`
- `scale/global_metrics.py`
- `scale1m/checkpoint.py`
- `scale1m/export_rf.py`
- `stage3HNSW/build_prior_sidecar.py`
- `scale1m/train_rung.py`
- `scale1m/a0_graph_validation.py`
- `scale1m/prepare_a0_graph.py`
- `scale1m/build_graph_rf.py`
- `scale1m/a0_recompute_checks.py`

Retargeting the formal-checkout junction to a verified immutable copy of all 155 A0.4 source files plus the historical English authority preserves these executable checks. Verify the 155-file set against `A0_4/A04_PREFLIGHT.json.source_sha256` and the 14 directly bound files against the original raw A0.6 manifest. Preserve the raw manifest and protocol bytes. Wait for the original A0.7 processes to finish before retargeting their input paths.

Continue invoking A0.7 independent recomputation from the **active local repository**, using the original A0.6 Linux protocol. `consume_source_counts` requires the resolved path and SHA of `a0_source_counts.py` to match the local producer implementation recorded in the source-count report. Invoking the reader from the newly frozen checkout instead would change the module's resolved path and can fail that identity check even when code bytes match.

The old Windows `A0_PROTOCOL.json` names the historical authority at the active document path. Keep it as a historical protocol. It is not the appropriate entry point for verifying the A0.6 raw manifest, whose protocol digest binds the Linux copy. Cite the Linux protocol and actual A0.7 as-run commands for this verification.

## 2. Append-only relocation record for existing report references

A0.7 reports record resolved paths in `artifact_hashes`. Before junction retargeting, the formal English authority resolves to the active repository's English path. Publishing a new active library changes that path's bytes, while the old report correctly retains its original digest.

Preserve the report bytes and add a separately hashed relocation document with entries containing:

```json
{
  "original_path": "D:/research/model_lake/codes/ModelLakeFishing/docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md",
  "expected_sha256": "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd",
  "relocated_path": "<verified immutable authority copy>",
  "verified_sha256": "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd",
  "reason": "Active evidence library revised after A0.7 verification; historical authority bytes preserved."
}
```

Apply relocation only to an exact original-path **and expected-digest** match. Preserve distinct before/after alias maps. Bind the original report SHA, relocation-record SHA, immutable snapshot manifest SHA, and new library SHAs in the revision manifest. Link this record in both revised libraries. Existing artifact hashes are historical verification records; they should not be silently rewritten to hashes of revised prose.

## 3. Stale citations to replace in both languages

Both pre-revision files contain the same 18 distinct other-series evidence links:

| Old citation | Appropriate A0-only replacement evidence |
|---|---|
| `F1_runs/PROVENANCE.json` | A0.1 snapshot/input audits and A0.7 source recount |
| `F15_runs/PROVENANCE.json` | A0.1 snapshot audit and A0.7 source recount |
| `F15_runs/f15b_coverage.json` | A0.7 source recount measurements |
| `F2_runs/F2_REPORT.json` | A0.7 canonicalization source recount |
| `F2_runs/F2_MERGE_REPORT.json` | A0.7 merge recount with all three canonical-output identity checks |
| `F2_runs/rf_gold_rules.json` | A0.1 bound rule artifact and A0.7 source recount |
| `F3_runs/LADDER_REPORT.json` | A0.1 row-map audit and A0.7 verified identities |
| `F4_runs/FEATS_REPORT.json` | A0.1 input audit, A0.3 repair audit, A0.7 feature recount |
| `F5_runs/GRAPH_REPORT.json` | A0.3 repaired graph and A0.7 graph checks |
| `F5_runs/f5_splits.json` | A0.1 query identity and A0.7 split/root checks |
| `F5_runs/lineage_stats.json` | A0.7 source recount and graph checks |
| `X4_runs/X4GD_full_s0_e25/metadata/resolved_config.json` | `A0_runs/A0_4/delivery_s0/metadata/resolved_config.json` |
| `X4_runs/X4GD_full_s1_e25/metadata/resolved_config.json` | `A0_runs/A0_4/delivery_s1/metadata/resolved_config.json` |
| `X4_runs/X4GD_full_s2_e25/metadata/resolved_config.json` | `A0_runs/A0_4/delivery_s2/metadata/resolved_config.json` |
| `X5_runs/X5_GD_ELIGIBILITY.json` | A0.7 independently recomputed metrics |
| `X6_runs/X6_BASELINES.training_free.json` | Remove baseline comparisons: A0 has no baseline rerun |
| `Y2_runs/Y2_REPORT.json` | A0.6 raw manifest/report and A0.7 independent report |
| `Y4_runs/Y4_REPORT.json` | A0.6 measured query timing and A0.7 recomputed quantiles |

There are currently no broken relative links in either pre-revision file; the problem is stale evidence scope, not link reachability. Code modules whose historical filenames contain a series identifier, such as `scale1m/eval_y2.py`, can remain implementation references when actually executed under the A0 protocol. The final method name `X4G+D` should remain.

## 4. Stale reproducibility and prose to replace

- Remove the entire optional `scale1m.eval_y4` sensitivity command block. A0 evaluates the fixed final top-1000 system and its two exact diagnostic references.
- Replace `X4GD_full_s<SEED>_e25`, `exports_x4`, `metrics_y2`, `metrics_y4`, and `hnsw_y2_eval.bin` result paths with actual A0 run/export/index locations.
- Do not present fresh hub crawling as reproducing the frozen A0 inputs. Preserve the A0.1 snapshot identities and recount those bound inputs.
- Training and export commands must use the repaired graph and A0 producer protocol; citing the recorded A0.4–A0.7 commands is safer than maintaining an incomplete manually reconstructed recipe.
- Replace old graph digest `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c` in claims about final A0 runs with repaired digest `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`.
- Replace pre-repair dataset statistics in the feature formula and method summary: seven performance-derived columns are zeroed while the 458-dimensional shape and remaining feature values are preserved.
- Replace every former headline/result/cost value, including 0.3031/30.31%, its seed-specific scores, full-pool reference, quality-retention percentages, timing and build durations. A0.7 values must be the current evidence.
- Remove the old BM25 superiority and 2.98-times claims. A0.7 provides no rerun of those comparison systems.
- Remove the old 3,004 API pages and zero skipped duplicates as current verified facts. A0.7 explicitly leaves these two original crawl-event counters missing.
- Preserve the evaluation scope: materialized dataset–task nodes, held-out performance edges, historical ranking recovery, fixed initialization and three root split seeds. The repaired seven feature columns resolve the particular identified feature leakage; this does not expand the evaluated task to unseen node encoding or downstream model execution.

## 5. Suggested parallel structure

Use identical section numbering and tables in both languages:

1. Current A0 evidence identity and sole final system.
2. Frozen data, source recount, row identities and feature repair.
3. Node features, encoder, training losses and frozen configuration.
4. Query/split scope, eligible counts, task-prior visibility and ranking metric definitions.
5. Final per-seed results and two exact diagnostic references.
6. HNSW calibration, latency, index and offline compute costs, with measurement scopes.
7. Correctness verification, artifact chain and actual reproduction entry points.
8. Evidence limitations and the two missing acquisition-event counters.

Retain technical detail needed to reproduce the method, but source every numerical fact from A0 artifacts. State that the metric inventory has 982 required items, with 838 recomputed, 86 verified, 3 disabled, 11 not applicable, 42 undefined and 2 missing if these counts remain the final A0.7 status. The report explicitly sets `complete=false` because of those two missing original event counters; distinguish that inventory status from passed retrieval computation and validation.

## 6. Publication gates for bilingual parity

- Compare all code tokens, formulas, numerical table cells, SHA values, metric IDs, paths and relative link targets between languages.
- Use the same units and aggregation definitions: per-seed scores, arithmetic mean across seeds, mean of per-seed retention ratios, and per-seed quantiles. Never label a mean of per-seed quantiles as a pooled-query quantile.
- Describe 4,122 as query observations across splits, not distinct dataset nodes.
- Distinguish formal remote retrieval timings from local A0.7 verification runtime.
- Verify every local link target, all equation dimensions, and all cited result table cells against the bound A0 JSON/CSV records.
- Search both revised documents for remaining other-series evidence links, old result paths and old graph digest.
- Hash both final files and archive the revision manifest after validation.

