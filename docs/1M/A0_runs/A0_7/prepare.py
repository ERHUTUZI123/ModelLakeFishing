from pathlib import Path
import json, hashlib, shutil, subprocess
BASE=Path(__file__).resolve().parent
DOC=BASE.parents[1]
ROOT=DOC.parents[1]
DATA=Path('D:/research/model_lake/data/data1m')
ALIAS=Path('D:/u801/x98liu/model_lake')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),(src,dst)
    else:shutil.copy2(src,dst)
frozen=BASE/'frozen'
for src in [DOC/'A0.md',DOC/'A0.6.md',DOC/'EVIDENCE_SOURCE_LIBRARY_en.md']:
    copy(src,frozen/src.name)
for name in ['A0_SOURCE_MANIFEST.json','A0_METRIC_INVENTORY.json','A0_OLD_REFERENCE.json','A0_COMMANDS.md','A0_DISCREPANCIES.md']:
    copy(BASE.parent/name,frozen/name)
aliases={ALIAS/'data1m':DATA,
         ALIAS/'a0_formal_20260913/ModelLakeFishing':ROOT,
         ALIAS/'a0_eval_20260914':BASE.parent/'A0_6'}
for seed in range(3):
    aliases[ALIAS/f'runs/A0_20260912/A0GD_full_s{seed}_e25']=BASE.parent/f'A0_4/delivery_s{seed}'
for link,target in aliases.items():
    assert target.is_dir(),target
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.exists():assert link.resolve()==target.resolve(),(link,target)
    else:
        command="New-Item -ItemType Junction -Path '"+str(link)+"' -Target '"+str(target)+"' | Out-Null"
        subprocess.run(['powershell','-NoProfile','-Command',command],check=True)
(BASE/'PATH_ALIASES.json').write_text(json.dumps({'scope':'Filesystem aliases only; all producer metadata, protocol and numerical artifact bytes preserved.',
    'aliases':{str(k):str(v) for k,v in aliases.items()}},indent=2)+'\n')
metrics=DATA/'a0_20260912/metrics'
for src in (BASE.parent/'A0_6/delivery/metrics').iterdir():
    if src.is_file():copy(src,metrics/src.name)
for seed in range(3):
    target=DATA/f'a0_20260912/exports/A0GD_full_s{seed}_e25'
    for src in (BASE.parent/f'A0_5/delivery/export_s{seed}').iterdir():
        if src.is_file():copy(src,target/src.name)
    copy(DATA/'a0_20260912/graph/unique_model_id.parquet',target/'model_ids.parquet')
required=json.loads((BASE/'required_files.json').read_text())
missing=[]
for remote,rec in required.items():
    p=Path(remote)
    if not p.exists():
        assert remote.startswith('/u801/x98liu/model_lake/data1m/a0_20260912/'),remote
        missing.append({'remote':remote,'local':str(p.resolve()),**rec})
    else:
        assert p.stat().st_size==rec['bytes'],p
download={'files':missing,'total_bytes':sum(r['bytes'] for r in missing)}
(BASE/'DOWNLOAD_PLAN.json').write_text(json.dumps(download,indent=2)+'\n')
print(json.dumps({'missing_files':len(missing),'download_GiB':download['total_bytes']/2**30,
                  'paths':[r['local'] for r in missing]}))
