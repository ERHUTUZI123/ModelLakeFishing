# Independent final QA of A0.7 backfill

Read-only audit of A0.7.md, A0.md section 10, A0_RESULTS.md, A07_VALIDATION.json and REPLAY_PATH_MAP.json. Findings reflect the files inspected at the 2026-09-15T01:35:52Z backfill. Active artifacts were not edited by this audit.

## Passed checks

- All 156 relocation records (155 source files and the historical English authority) match both their expected SHA256 and recorded verified SHA256.
- The formal-checkout compatibility junction resolves to the immutable checkout named in REPLAY_PATH_MAP.json.
- A07_VALIDATION.json binds the current original A0_REPORT.json and A0_METRIC_INVENTORY.json hashes correctly.
- All 470 additional comparison names are distinct; their count agrees with the validation summary.
- All relative links in A0.7.md, A0.md and A0_RESULTS.md resolve.
- The final per-seed gold@10 values, three-seed mean and displayed percentage-point changes agree with the A0.7 report.
- The 982-item inventory counts sum correctly: 838 recomputed, 86 verified, 3 disabled, 11 not applicable, 42 undefined and 2 missing. The two missing IDs are the original snapshot API-page and discarded-duplicate event counts; no seeds or invalid values are missing.
- The new per-seed and mean latency values, per-index mean byte count, and A0.md total-index-size presentation agree with the A0.7 report.
- STATUS.json explicitly records the interrupted supervisor's unknown OS exit code and endpoint. The report-derived expected CLI exit code is labelled as expected, not observed. The source-count supervisor interval is explicitly distinguished from isolated recount duration.
- The results text correctly states that both exact-reference timing entries share one scan and should be counted once, and distinguishes the remote formal timing environment from local independent verification.

## Corrections required for clear evidence presentation

1. **Mixed ratio and percentage units in A0_RESULTS.md, retention/coverage table.** New entries such as 0.9353648304 are ratios; the old entry 94.32 is a percent value under the same unqualified column label. Render consistent units or explicitly annotate old values with `%` and explain the new ratio scale. Apply this to all five rows.

2. **Conflated absent/undefined/not-applicable states in supplementary tables.** The blanket text `未报告 / 无定义` is not a faithful description of distinct states. Missing historical fields should read `未报告`; selected-ef and calibration-pass aggregate columns should read `不适用` when no numerical mean is defined for the reporting contract. Reserve `无定义` for genuinely mathematical undefined measurements with an explicit reason.

3. **Stale mapping description in A0.7.md.** PATH_ALIASES.json is the original local execution mapping. The formal checkout now uses REPLAY_PATH_MAP.json. Link both and distinguish their historical/current roles so the reproduction entry point does not send readers to the old mutable authority.

4. **Stale next-step entries in A0.md section 10.2.** Rows for completed stages still state A0.4 pending or request already completed training/export acceptance. Refresh those current-status cells while retaining historical log entries in section 10.6.

5. **Three-index total should be explicit in the full results cost section.** The shown per-index mean is correct. Add total `7,133,409,984` bytes / `6.6435057521` GiB under a sum-labelled field; do not place the sum in a mean column. A0.md already shows the correct total.

No evidence from this QA requires rerunning the numerical pipeline. The required corrections concern presentation and replay documentation; original A0_REPORT and source-count bytes should remain preserved.

