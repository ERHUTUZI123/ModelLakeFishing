set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
sacct -j 1539529,1539530 --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P > sacct_final.txt
python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path.cwd(); run=Path('/u801/x98liu/model_lake/runs/A0_20260912/smoke_s0_e1_retry1')
report=json.loads((run/'SMOKE_REPORT.json').read_text()); resources=json.loads((root/'observations/RESOURCE_REPORT.json').read_text())
cfg=json.loads((run/'metadata/resolved_config.json').read_text())['resolved_config']
expected=json.loads((root/'ModelLakeFishing/docs/1M/A0_runs/A0_PROTOCOL.json').read_text())['required_config_audit']['resolved_config']
assert cfg==expected and len(cfg)==40
assert resources['returncode']==0
assert report['status']=='PASS' and report['mechanism_gate']['passed']
assert report['mechanism_gate']['checkpoint_roundtrip']
assert report['epochs']==1 and report['start_epoch']==0 and report['seed']==0
assert not report['evaluation_performed'] and not report['formal_training_eligible']
assert report['graph_sha256']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for rel,digest in manifest['sources'].items():
    if rel!='a0_ops/smoke.sbatch':assert sha(root/'ModelLakeFishing'/rel)==digest,rel
validation={'status':'PASS','job_id':1539530,'config_keys_equal':40,'graph_sha256':report['graph_sha256'],'source_files_unchanged_except_environment_script':len(manifest['sources'])-1,'smoke_gate':report['mechanism_gate'],'formal_training_started':False,'resource_report':resources}
(root/'REMOTE_SMOKE_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
files=[p for p in run.rglob('*') if p.is_file()]
run_hashes={str(p.relative_to(run)):sha(p) for p in files}
(root/'RUN_FILE_HASHES.json').write_text(json.dumps(run_hashes,indent=2)+'\n')
with tarfile.open(root/'smoke_delivery.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname='run/'+str(p.relative_to(run)))
    for p in root.glob('*.json'):tar.add(p,arcname=p.name)
    for p in root.glob('*.txt'):tar.add(p,arcname=p.name)
    for p in root.glob('*.log'):tar.add(p,arcname=p.name)
    for p in (root/'observations').iterdir():tar.add(p,arcname='observations/'+p.name)
    tar.add(root/'ModelLakeFishing/a0_ops/smoke.sbatch',arcname='executed_smoke.sbatch')
print(json.dumps({'status':'PASS','delivery_bytes':(root/'smoke_delivery.tar.gz').stat().st_size,'delivery_sha256':sha(root/'smoke_delivery.tar.gz'),'resource_report':resources,'gate':report['mechanism_gate']}))
PY
cat sacct_final.txt
