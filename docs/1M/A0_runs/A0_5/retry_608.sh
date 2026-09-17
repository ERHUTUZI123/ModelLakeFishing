set -eu
cd /u801/x98liu/model_lake/a0_export_20260913
python3 - <<'PY'
from pathlib import Path
import json
base=Path.cwd()
for seed in range(3):assert not (base.parent/f'data1m/a0_20260912/exports/A0GD_full_s{seed}_e25').exists()
s=Path('run_any_gpu.sbatch').read_text()
s=s.replace('#SBATCH --gres=gpu:1','#SBATCH --gres=gpu:1\n#SBATCH --nodelist=watgpu608')
s=s.replace('python -m pip freeze >', '''python -c "import sys;print('[python]',sys.version,sys.executable);assert sys.version_info[:2]==(3,11),'Staged environment requires Python 3.11'"
python -m pip freeze >''')
Path('run_608.sbatch').write_text(s)
Path('previous_job_1540850.txt').write_text(Path('job_id.txt').read_text())
PY
sbatch --parsable run_608.sbatch | tee job_id.txt
squeue -u x98liu
