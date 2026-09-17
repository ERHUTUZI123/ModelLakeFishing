set -eu
cd /u801/x98liu/model_lake/a0_export_20260913
state=$(squeue -h -j 1540848 -o '%T')
test "$state" = PENDING
scancel 1540848
python3 - <<'PY'
from pathlib import Path
p=Path('run.sbatch');s=p.read_text()
s=s.replace('#SBATCH --nodelist=watgpu308\n','').replace('#SBATCH --gres=gpu:schoolgpu:1','#SBATCH --gres=gpu:1')
Path('run_any_gpu.sbatch').write_text(s)
PY
sbatch --parsable run_any_gpu.sbatch | tee job_id.txt
squeue -u x98liu
