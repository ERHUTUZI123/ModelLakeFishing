set -eu
cd /u801/x98liu/model_lake/a0_formal_20260913
python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil
root=Path.cwd(); old=root.parent/'a0_smoke_20260913'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((old/'BUNDLE_MANIFEST.json').read_text())
for rel,digest in manifest['sources'].items():
    if rel!='a0_ops/smoke.sbatch':assert sha(old/'ModelLakeFishing'/rel)==digest,rel
smoke=json.loads((root.parent/'runs/A0_20260912/smoke_s0_e1_retry1/SMOKE_REPORT.json').read_text())
assert smoke['status']=='PASS' and smoke['mechanism_gate']['checkpoint_roundtrip']
for seed in range(3):assert not (root.parent/f'runs/A0_20260912/A0GD_full_s{seed}_e25').exists()
repo=root/'ModelLakeFishing';repo.mkdir()
sources={}
for rel,digest in manifest['sources'].items():
    if rel.startswith('a0_ops/'):continue
    dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(old/'ModelLakeFishing'/rel,dest)
    assert sha(dest)==digest,rel
    sources[rel]=digest
write={'status':'PASS','source_sha256':sources,'smoke_job_id':1539530,'smoke_report_sha256':sha(root.parent/'runs/A0_20260912/smoke_s0_e1_retry1/SMOKE_REPORT.json'),'graph_sha256':smoke['graph_sha256'],'family_vocab_sha256':manifest['family_vocab_sha256'],'formal_directories_all_absent':True,'ops_sha256':{p.name:sha(p) for p in root.iterdir() if p.is_file()}}
(root/'A04_PREFLIGHT.json').write_text(json.dumps(write,indent=2)+'\n')
print('PASS: smoke, source identities and fresh directories verified')
PY
sbatch --parsable train.sbatch | tee array_job_id.txt
squeue -u x98liu
