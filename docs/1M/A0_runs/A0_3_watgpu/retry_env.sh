set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
cp ModelLakeFishing/a0_ops/smoke.sbatch smoke_job1539529.sbatch
python3 - <<'PY'
from pathlib import Path
p=Path('ModelLakeFishing/a0_ops/smoke.sbatch')
s=p.read_text()
s=s.replace('python -m pip freeze > ../pip_freeze.txt','python -m pip install pytest==9.1.1\npython -m pip freeze > ../pip_freeze.txt')
p.write_text(s)
PY
sha256sum ModelLakeFishing/a0_ops/smoke.sbatch > smoke_retry_sha256.txt
sbatch --parsable ModelLakeFishing/a0_ops/smoke.sbatch | tee job_id_retry.txt
squeue -u x98liu
