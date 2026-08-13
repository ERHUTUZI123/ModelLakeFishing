#!/bin/bash
# Read-only capacity/exec probe for choosing a watGPU staging filesystem.

#SBATCH --job-name=mlf-t6-storage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail
: "${ENV_SOURCE:?set ENV_SOURCE}"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
du -sh "${ENV_SOURCE}" || true

for candidate in "${SLURM_TMPDIR:-}" /tmp /var/tmp /dev/shm /local /localscratch /scratch; do
  [[ -n "${candidate}" && -d "${candidate}" ]] || continue
  echo "=== ${candidate} ==="
  df -h "${candidate}" || true
  findmnt -T "${candidate}" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
  test_dir=$(mktemp -d "${candidate%/}/mlf-exec-${USER}-${SLURM_JOB_ID}.XXXXXX") || continue
  cp /bin/true "${test_dir}/true"
  chmod 700 "${test_dir}/true" || true
  if "${test_dir}/true"; then
    echo "exec_test=PASS"
  else
    echo "exec_test=FAIL"
  fi
  rm -f "${test_dir}/true"
  rmdir "${test_dir}" || true
done
