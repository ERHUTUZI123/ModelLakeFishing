#!/bin/bash
# Shared watGPU environment staging helpers.  The NFS home is noexec, while
# node-local capacity differs across hosts, so select a writable executable
# filesystem with enough free space at job runtime.

mlf_choose_stage_base() {
  local env_source=$1
  local env_kb needed_kb candidate available_kb test_dir
  env_kb=$(du -sk "${env_source}" | awk '{print $1}')
  # Leave 2 GiB for transient pip files, bytecode, and filesystem variance.
  needed_kb=$((env_kb + 2 * 1024 * 1024))

  declare -A seen=()
  for candidate in "${SLURM_TMPDIR:-}" /dev/shm /tmp /var/tmp; do
    [[ -n "${candidate}" && -d "${candidate}" && -w "${candidate}" ]] || continue
    [[ -z "${seen[${candidate}]:-}" ]] || continue
    seen[${candidate}]=1
    available_kb=$(df -k --output=avail "${candidate}" 2>/dev/null | tail -1 | tr -d ' ')
    [[ "${available_kb}" =~ ^[0-9]+$ ]] || continue
    (( available_kb >= needed_kb )) || {
      echo "[env] skip ${candidate}: available=${available_kb}K needed=${needed_kb}K" >&2
      continue
    }
    test_dir=$(mktemp -d "${candidate%/}/mlf-exec-${USER}-${SLURM_JOB_ID}.XXXXXX") || continue
    cp /bin/true "${test_dir}/true" 2>/dev/null || { rm -rf -- "${test_dir}"; continue; }
    chmod 700 "${test_dir}/true" 2>/dev/null || true
    # A Python venv also needs archive metadata/symlink semantics.  Reject a
    # filesystem where the same cp mode used for staging is unsupported.
    if "${test_dir}/true" 2>/dev/null \
        && cp -a "${env_source}/pyvenv.cfg" "${test_dir}/pyvenv.cfg" 2>/dev/null \
        && cp -a "${env_source}/bin/python3" "${test_dir}/python3" 2>/dev/null; then
      rm -rf -- "${test_dir}"
      echo "[env] selected ${candidate}: available=${available_kb}K env=${env_kb}K" >&2
      printf '%s\n' "${candidate}"
      return 0
    fi
    rm -rf -- "${test_dir}"
  done
  echo "[fail] no writable executable staging filesystem has ${needed_kb}K free" >&2
  return 1
}

mlf_copy_env() {
  local source=$1 destination=$2
  mkdir -p "${destination}"
  # Copy contents rather than a possible top-level symlink, preserving the
  # venv metadata and internal links exactly as the T4/T5 staging contract.
  cp -a "${source}/." "${destination}/"
}
