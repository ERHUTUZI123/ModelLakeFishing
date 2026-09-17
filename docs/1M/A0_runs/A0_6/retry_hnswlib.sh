set -eu
cd /u801/x98liu/model_lake/a0_eval_20260914
test ! -e /u801/x98liu/model_lake/data1m/a0_20260912/metrics
python3 - <<'PY'
from pathlib import Path
s=Path('run_308.sbatch').read_text()
s=s.replace('python -c "import sys,torch,hnswlib;', 'python -m pip install hnswlib==0.8.0\npython -c "import sys,torch,hnswlib;')
Path('run_308_hnswlib.sbatch').write_text(s)
Path('job_id_before_hnswlib.txt').write_text(Path('job_id.txt').read_text())
PY
sbatch --parsable run_308_hnswlib.sbatch | tee job_id.txt
squeue -u x98liu
