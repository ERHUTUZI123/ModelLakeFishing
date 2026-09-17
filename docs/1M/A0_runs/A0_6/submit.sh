set -eu
cd /u801/x98liu/model_lake/a0_eval_20260914
echo '9018885792a35f09dba893b66f5922997acccc46439453b28e311c2fde80c111  inputs.tar.gz' | sha256sum -c -
tar -xzf inputs.tar.gz
python3 - <<'PY'
from pathlib import Path
import hashlib,json
base=Path.cwd();project=base.parent/'a0_formal_20260913'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for rel,digest in json.loads((project/'A04_PREFLIGHT.json').read_text())['source_sha256'].items():assert sha(project/'ModelLakeFishing'/rel)==digest,rel
for seed in range(3):
    proof=json.loads((base.parent/f'a0_export_20260913/A05_VALIDATION_s{seed}.json').read_text())
    assert proof['status']=='PASS'
    ex=base.parent/f'data1m/a0_20260912/exports/A0GD_full_s{seed}_e25'
    for name,rec in proof['files'].items():assert sha(ex/name)==rec['sha256'],name
p=json.loads((base/'inputs/A0_PROTOCOL.linux.json').read_text())
for r in [p['authority'],p['input_audit_binding']['input_audit'],p['input_audit_binding']['query_identity']]:assert sha(Path(r['path']))==r['sha256']
audit=json.loads(Path(p['input_audit_binding']['input_audit']['path']).read_text())
original=next(r for r in audit['files'] if Path(r['path']).name=='x_dataset.npy')
assert sha(Path(original['path']))==original['sha256']
assert not (base.parent/'data1m/a0_20260912/metrics').exists()
(base/'PREFLIGHT.json').write_text(json.dumps({'status':'PASS','A05_artifacts_all_hashed':True,'source_code_matches_A04':True,'fresh_metrics_directory':True,'original_dataset_feature_verified':original,'path_relocation':json.loads((base/'PATH_RELOCATION.json').read_text()),'operations_sha256':{p.name:sha(p) for p in base.iterdir() if p.is_file()}},indent=2)+'\n')
print('PASS: A0.5 actual artifact hashes, frozen sources, original features, Linux path copies, fresh output')
PY
sbatch --parsable run.sbatch | tee job_id.txt
squeue -u x98liu
