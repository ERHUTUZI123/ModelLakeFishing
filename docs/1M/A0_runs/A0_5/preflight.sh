set -eu
squeue -u x98liu
python3 - <<'PY'
from pathlib import Path
import json,hashlib
base=Path('/u801/x98liu/model_lake');formal=base/'a0_formal_20260913'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for rel,digest in json.loads((formal/'A04_PREFLIGHT.json').read_text())['source_sha256'].items():assert sha(formal/'ModelLakeFishing'/rel)==digest,rel
for name,digest in [('ladder_rf/full_model_ids.parquet','fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa'),('rf/canon/dataset_nodes_merged.parquet','763201d8e6e103a664b658ec7148b657d28f1d0cdc6f70a54b06687122dbdc57')]:
    assert sha(base/'data1m'/name)==digest,name
for seed in range(3):
    proof=json.loads((formal/f'observations_s{seed}/A04_VALIDATION.json').read_text())
    assert proof['status']=='PASS' and proof['epochs']==25 and proof['checkpoint_epoch']==24
    assert sha(base/f'runs/A0_20260912/A0GD_full_s{seed}_e25/ckpt/last.pt')==proof['checkpoint_sha256']
    assert not (base/f'data1m/a0_20260912/exports/A0GD_full_s{seed}_e25').exists()
print('PASS: frozen source, three final checkpoints, ladder, task-nodes and fresh export directories')
PY
