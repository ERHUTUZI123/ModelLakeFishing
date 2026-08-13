#!/bin/bash
# Final machine-readable and human-readable summary for the three full seeds.

#SBATCH --job-name=mlf-t6-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
: "${T6_CONFIG:?set T6_CONFIG}"
# shellcheck disable=SC1090
source "${T6_CONFIG}"

export T6_STATE_DIR
export T6_ALLOW_NO_GOLD_GATE
python3 - <<'PY'
import json
import os
import sys

state = os.environ["T6_STATE_DIR"]
stages = ["12k", "30k", "100k_smoke", "100k_s0", "100k_s1", "100k_s2"]
reports = {}
errors = []
for stage in stages:
    path = os.path.join(state, "reports", f"gate_{stage}.json")
    if not os.path.isfile(path):
        errors.append(f"missing report: {stage}")
        continue
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    reports[stage] = report
    if not report.get("passed"):
        errors.append(f"failed gate: {stage}: {report.get('errors')}")

override = os.environ.get("T6_ALLOW_NO_GOLD_GATE", "0") == "1"
summary = {
    "passed": not errors,
    "errors": errors,
    "reports": reports,
    "gold10_validated": False,
    # The opt-in has to be legible here, not only in jobs.tsv: SUMMARY.json is
    # what a later reader trusts, and "passed: true" without this field reads
    # like the gold gate was satisfied rather than deliberately deferred.
    "gold_gate_override": override,
    "scientific_status": (
        "T6 training complete; T7/T8 gold@10 validation pending"
        + (" (advanced under T6_ALLOW_NO_GOLD_GATE=1)" if override else "")
    ),
}
path = os.path.join(state, "SUMMARY.json")
with open(path + ".tmp", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(path + ".tmp", path)
with open(os.path.join(state, "STATUS"), "w", encoding="utf-8") as fh:
    fh.write("TRAINING_COMPLETE\n" if summary["passed"] else "FAILED_FULL_SEEDS\n")
print(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(0 if summary["passed"] else 1)
PY
