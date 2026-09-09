#!/bin/bash
# Collect the T6 evidence required by docs/1M/T6GPU.md without downloading the
# large model/optimizer checkpoints.  Checkpoints remain in OUTPUT_ROOT/runs.

set -euo pipefail

: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${DATA_ROOT:?set DATA_ROOT}"

workflow_dir=${1:?usage: collect_t6_artifacts.sh WORKFLOW_DIR [ARCHIVE]}
archive=${2:-"${HOME}/$(basename "${workflow_dir}")_delivery.tgz"}
[[ -d "${workflow_dir}" ]] || { echo "[fail] no workflow: ${workflow_dir}" >&2; exit 2; }
[[ "$(cat "${workflow_dir}/STATUS")" == TRAINING_COMPLETE ]] || {
  echo "[fail] workflow is not TRAINING_COMPLETE" >&2
  exit 2
}

delivery="${workflow_dir}/delivery"
mkdir -p "${delivery}"

job_ids=$(awk -F '\t' '$2 != "meta" {print $3}' "${workflow_dir}/jobs.tsv" | paste -sd, -)
for attempt in 1 2 3; do
  if sacct -j "${job_ids}" \
      --format=JobID,JobName,State,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode \
      > "${delivery}/sacct_all_jobs.txt"; then
    break
  fi
  [[ "${attempt}" == 3 ]] && exit 1
  sleep 2
done

{
  echo -e "run_id\tpath\tbytes"
  for run_id in \
    R2_12k_s0_e25 R2_30k_s0_e25 R2_100k_s0_e2 \
    R2_100k_s0_e25 R2_100k_s1_e25 R2_100k_s2_e25; do
    run_dir="${OUTPUT_ROOT}/runs/${run_id}"
    for name in ckpt/last.pt ckpt/best.pt ckpt/family_vocab.csv; do
      path="${run_dir}/${name}"
      [[ -e "${path}" ]] || continue
      printf '%s\t%s\t%s\n' "${run_id}" "${path}" "$(stat -Lc %s "${path}")"
    done
  done
} > "${delivery}/checkpoint_inventory.tsv"

{
  for graph in \
    "${DATA_ROOT}/data1m/graphs/hgraph_12k.pt" \
    "${DATA_ROOT}/data1m/graphs/hgraph_30k.pt" \
    "${DATA_ROOT}/data1m/graphs/hgraph_100k.pt"; do
    sha256sum "${graph}"
  done
} > "${delivery}/graph_sha256.txt"

sha256sum "${workflow_dir}"/artifacts/*.tgz \
  > "${delivery}/artifact_sha256.txt"

tar -czf "${archive}" -C "${workflow_dir}" \
  STATUS jobs.tsv SUMMARY.json workflow.env runtime.env \
  reports artifacts logs delivery
sha256sum "${archive}" > "${archive}.sha256"

echo "[done] ${archive}"
echo "[done] ${archive}.sha256"
