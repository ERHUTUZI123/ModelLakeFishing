# Independent numerical audit for the bilingual evidence libraries

Scope: only A0.1–A0.7 evidence and the current libraries as the list of claims to replace. No other result series was read. `audit_numeric_check.py` independently reads A0 raw NPZ arrays and index file sizes; all 42 checks passed in `audit_numeric_check.json`.

## Authoritative current result

Source: `../results/A0_REPORT.json`, `new_measurements.<metric>.<seed|mean>.value`; all values below are taken from new measurements, not `old_reference` or `comparisons`. Raw arrays: `D:/research/model_lake/data/data1m/a0_20260912/metrics/a0_hnsw_s{seed}.npz` and `a0_exact_s{seed}.npz`.

Sole final system: **X4G+D → HNSW top1000 → task prior → top10** (`hnsw1000_task_prior`). Exact pools and exact full-lake fusion are diagnostic references for this same trained representation/fusion. A0 contains no rerun BM25 baseline; remove old superiority/2.98× baseline claims from the current fact libraries.

| Metric | seed 0 | seed 1 | seed 2 | Arithmetic mean across seeds |
|---|---:|---:|---:|---:|
| Final gold@1 | 0.1585365854 | 0.1217075386 | 0.1022653722 | 0.1275031654 |
| Final gold@10 | 0.3319783198 | 0.3369663942 | 0.2213592233 | 0.2967679791 |
| Final top3@10 | 0.3739837398 | 0.4096276113 | 0.2627831715 | 0.3487981742 |
| Final gold-gap@10 | 0.3550135501 | 0.3660308810 | 0.2310679612 | 0.3173707974 |
| Final root-macro gold@10 | 0.2515496787 | 0.2566173016 | 0.2470558235 | 0.2517409346 |
| Conditional median gold position, when in pool | 4 | 5 | 4 | Keep seed medians; 4.3333 is merely mean of medians |
| Exact top1000 + prior gold@10 | 0.3326558266 | 0.3369663942 | 0.2213592233 | 0.2969938147 |
| Exact full-lake + prior gold@10 | 0.3604336043 | 0.3542234332 | 0.2375404531 | 0.3173991636 |
| Exact-pool retention ratio | 0.9229323308 | 0.9512820513 | 0.9318801090 | 0.9353648304 |
| ANN retention ratio | 0.9979633401 | 1 | 1 | 0.9993211134 |
| Overall retention ratio | 0.9210526316 | 0.9512820513 | 0.9318801090 | 0.9347382640 |
| Exact top1000 gold coverage | 0.5386178862 | 0.5740236149 | 0.3708737864 | 0.4945050958 |
| HNSW top1000 gold coverage | 0.5392953930 | 0.5740236149 | 0.3708737864 | 0.4947309314 |
| Full-fused top10 members in exact dense pool | 0.8815040650 | 0.9070844687 | 0.9211650485 | 0.9032511941 |

Retention aggregation is **mean of per-seed ratios**, not ratio of means. Root-macro first averages query indicators within each root, then averages roots, then the displayed overall row averages the three seed metrics. Conditional median is based on gold being present anywhere in the 1000-candidate pool, not only successful top10 queries. Coverage and retention are distinct quantities.

## Queries, ANN, latency and index cost

| Quantity | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Query observations | 1476 | 1101 | 1545 |
| Distinct roots within that split | 730 | 495 | 473 |
| Selected ef_search | 1000 | 1500 | 1000 |
| Selected recall@1000 | 0.994381436314363 | 0.9962606721162578 | 0.992820064724919 |
| HNSW p50 (ms) | 0.512913 | 0.716782 | 0.536364 |
| HNSW p95 (ms) | 0.8041555 | 1.148937 | 0.7549026 |
| Prior rerank p50 (ms) | 0.178343 | 0.169012 | 0.124475 |
| Prior rerank p95 (ms) | 0.22924075 | 0.209267 | 0.2084604 |
| Combined p50 (ms) | 0.6934055 | 0.882341 | 0.666437 |
| Combined p95 (ms) | 1.0143045 | 1.337841 | 0.9531152 |
| Index build seconds | 196.151983493 | 208.427163905 | 217.314887888 |
| Index bytes | 2377803152 | 2377803548 | 2377803284 |

Combined mean-of-seed p50 = 0.7473945 ms, p95 = 1.1017535667 ms; component mean p50 values are 0.5886863333 and 0.1572766667 ms. Combined percentiles come from per-query `total_ns = hnsw_ns + rerank_ns`; they are not the sum of component percentiles. The three index files total 7,133,409,984 bytes = 6.6435057521 GiB. Index byte counts are disk sizes, not memory peaks.

4,122 is the number of query observations across the three splits; do not call it the number of distinct dataset–task queries. Each split uses N = 3,016,439 candidates, K = 1,000 and returns 10. Seed 1 ef=1000 recall was 0.9897847411, hence the next ef=1500 was used. Other untried ef-grid entries are `not_applicable` after first-passing stop, not failures or measured zero.

Source: `A0_EVALUATION_MANIFEST.json#/binding/settings` records `timed_passes=1`, warmup `min(50,Q)`, M32, ef_construction200, build threads8, query_chunk16, model_chunk50000. Therefore replace any old “repeated timing” claim with “one timed pass per seed.” Formal timing is remote watgpu308/Linux; 4060+i7 performed additional local verification. Online timing starts at a materialized query vector and excludes query encoding and index load/build. HNSW query timing is single-threaded; build uses eight threads.

## Other costs

Source: `A0_REPORT.json#/new_measurements/pipeline_cost.*`; producer scope is preserved in `A0_EVAL_RECORDS_s{seed}.json`.

| Seconds | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Training | 3875.623107513 | 3911.995430701 | 3569.225913043 |
| Export | 36.909912889 | 39.057959747 | 40.113738693 |
| Prior build | 7.548296398 | 7.497223104 | 7.522241180 |
| Shared exact scan (both exact references) | 8.752328499 | 6.686751143 | 7.996605582 |
| Complete per-seed evaluation segments | 210.592887219 | 220.728925835 | 231.004162807 |

Exact full-lake and exact top1000 reference times describe the SAME scan. Complete evaluation includes exact + HNSW (including index build) once and excludes shared binding/finalize and idle gaps. Thus none of these overlapping rows may be blindly summed. Process RSS peaks include lifetime history/earlier seeds; evaluator GPU peaks use torch max allocated without per-seed reset. Do not label these isolated per-seed resource footprints or overall node memory usage. Runtime environments in `reproducibility.runtime_environment.*` describe training on watgpu308 L40S; export occurred on watgpu608 RTX6000Ada, and formal exact/HNSW evaluation on watgpu308.

## Completeness and source counts

982 inventory items: 838 recomputed, 86 verified, 42 mathematically undefined, 11 not applicable, 3 disabled, 2 missing; zero invalid; all three seeds present; all ANN fidelity gates pass. **A0 all-metric completeness remains false** because the original model-crawl API-page event log and duplicate-discard event log are unavailable. Remove “3004 API pages” and “zero skipped duplicates” as current facts. Retained unique/shard counts are independently supported; unique retained records cannot recover discarded duplicate occurrences or page boundaries.

The fresh A0 source recount supports unchanged values: model snapshot 3,003,759 in 61 shards; dataset snapshot 1,008,417 in 11 shards; native raw metric rows 2,158,375, finite 2,097,081, median-deduplicated 1,435,162, primary edges before/after cap 143,478 / 74,346; six-source input 531,958, within-source duplicates removed 5,102, cross-source conflicts 1,502, unique uncapped pairs 525,354, retained 247,803. Retained source counts 117,898 / 73,670 / 45,992 / 5,223 / 3,670 / 1,350 (named sources as in A0_SOURCE_COUNTS, sum247,803).

Frozen graph dimensions: 3,016,439 model rows, 18,729 dataset–task nodes; input widths448/458, output128; model feature file5,405,458,816 bytes (5.405458816 decimal GB). Family lookup index cardinality41,056 versus40,927 actually used IDs; these are different quantities. Unknown-size2,169,706/3,016,439=71.92938428%; Other-family409,057/3,016,439=13.56092399%; matched cards3928/18729=20.97282289%. Stored relationcounts247803/247803/374580/859065/859065 total2588316; top10-unweighted training similarity187290.

The seven repaired performance-derived dataset-feature columns must be described as zero across all nodes; new graph digest `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`. Preserve 458-wide schema with 451 other columns byte-identical. The old unrepaired graph digest is provenance only, not the graph used for current results.

## Reporting-unit correction

The independent provenance review identified a reporting bug in the original A0.7 output: six ratio/coverage metrics had native fractional values but the inventory declared `percent`. The original comparison then subtracted the historical percent number from a fractional new number. This affected reporting/old-new differences, while all raw retrieval numbers and the 42 independent numerical checks above remained correct.

At the parent's request, the active `scale1m/recompute_a0.py` now retains fractional scientific values with explicit `unit="fraction"`, propagates that unit through seed summaries, converts once into inventory `percent`, and labels comparison deltas `percentage_points`. Already-percent source statistics receive `unit="percent"` and scale1. Missing/undefined values stay null. The renderer labels percent rows and report provenance records SHA256 of both active independent readers. The original report/inventory/Markdown and 155 original producer-code files were preserved by the parent before this edit.

Validation: `scale1m/tests/test_recompute_a0.py`: **39 passed**. After the final explicit source-unit assertion, the filtered rerun passed all seven selected cases (six newly added cases plus the existing timing-percentiles case). New coverage includes frozen historical `94.32`, unequal seed ratios (mean of ratios rather than ratio of sums), six ratios from the raw pool fixture, already-percent source statistics with/without a preexisting tag, missing/undefined results, unchanged frozen inventory bytes, and active independent-reader hashes. Updated reader SHA256: `3ce80ac6f066716bef0f8a7731ad6abcb25fb94d3f79f7d9cf9ce5b36cf0c0f9`. Parent will regenerate the current A0.7 outputs with this reporting fix; original evaluator/data bytes and all raw scientific values are preserved.
