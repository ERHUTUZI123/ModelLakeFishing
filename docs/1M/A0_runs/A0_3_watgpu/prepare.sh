set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
echo '272363cee278f120815c41912978c86c5f3e88528021b33c01a60e13556488e2  a03_smoke_bundle.tar.gz' | sha256sum -c -
tar -xzf a03_smoke_bundle.tar.gz
python3 prepare_remote.py
printf '\nSCHEDULING\n'
sbatch --parsable ModelLakeFishing/a0_ops/smoke.sbatch | tee job_id.txt
squeue -u x98liu
