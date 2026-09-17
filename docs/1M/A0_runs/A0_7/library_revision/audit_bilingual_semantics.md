# Bilingual numerical and measurement-scope review

Independent EN table validation checks146 measured cells against the current A0.7 report and A0.1 snapshot audit. The bilingual table comparison checks171 numerical cells across14 corresponding tables. Only Chinese punctuation in the configuration seed list is localized; values, percentages, decimal precision and measured units agree.

Manual EN/ZH review confirms:

- The sole final system is GD representation + HNSW1000 + fixed task prior → top10; the exact rows are diagnostic references for the same representation/fusion.
- Query counts1476/1101/1545 and per-split root counts730/495/473 are distinguished. Summed4122 counts split observations and allows nodes to recur across splits.
- Quality is an equally weighted mean of three seed metrics. Root-macro averages within roots before averaging roots. Retention averages per-seed ratios, with percent display explicitly converted from native fractions.
- The conditional median4/5/4 is based on gold entering the candidate pool, supported by the preceding metric-definition paragraph; it is not a full-lake median or a median limited to successful top10 queries.
- ANN recall is candidate-ID overlap with exact dense1000, calibrated using the first passing ef under the original grid; seed2 uses ef1000. Gold labels are excluded from calibration.
- Online cost describes one timed pass per seed after warmup, from precomputed query vectors, with single-thread HNSW queries. Means of seed percentiles are distinguished from pooled percentiles and sums of component percentiles.
- Recorded exact references share one scan. Recorded exact+HNSW evaluation includes index build, so overlapping cost rows are not added twice. Disk GiB, process lifetime RSS high-water and PyTorch allocated-memory peaks have distinct scopes. Remote formal timing is distinct from local4060/i7 verification.
- Retrieval results are validated while overall982-item completeness staysfalse for two original crawl-event logs; undefined and disabled items are not interpreted as missing measurements or zeros.
- Historical observed-ranking recovery, frozen node membership, transductive information scope and absence of a validated new-node path are equivalent in both versions.

The final author saves resolve the requested wording clarification: §7/§8 now say the seven performance-derived columns are **zeroed / 归零**, retaining the458-column schema. The feature section explicitly states zero-based column indices and the surviving log-transformed root count. Final locked drafts passed both numerical scripts again; file hashes are captured by `audit_en_numeric.json` and `audit_bilingual_numeric.json`. Review status: **PASS**, with no unresolved numerical or measurement-scope finding.
