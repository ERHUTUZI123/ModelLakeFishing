set -eu
squeue -u x98liu
scontrol show node watgpu308
scontrol show node watgpu608
df -h /u801/x98liu/model_lake
python3 - <<'PY'
from pathlib import Path
import json
root=Path('/u801/x98liu/model_lake')
smoke=json.loads((root/'runs/A0_20260912/smoke_s0_e1_retry1/SMOKE_REPORT.json').read_text())
assert smoke['status']=='PASS' and smoke['mechanism_gate']['passed']
for seed in range(3):
    run=root/f'runs/A0_20260912/A0GD_full_s{seed}_e25'
    print(str(run),'exists=',run.exists())
    assert not run.exists(),run
print('SMOKE PASS; all three formal run directories absent')
PY
