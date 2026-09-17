set -eu
cd /u801/x98liu/model_lake/a0_eval_20260914
test "$(squeue -h -j 1540916 -o '%T')" = PENDING
test ! -e /u801/x98liu/model_lake/data1m/a0_20260912/metrics
scancel 1540916
python3 - <<'PY'
from pathlib import Path
p=Path('run.sbatch');s=p.read_text().replace('#SBATCH --nodelist=watgpu608','#SBATCH --nodelist=watgpu308')
Path('run_308.sbatch').write_text(s)
Path('job_id_608.txt').write_text(Path('job_id.txt').read_text())
PY
sbatch --parsable run_308.sbatch | tee job_id.txt
squeue -u x98liu
