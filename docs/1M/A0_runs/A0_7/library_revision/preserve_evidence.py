"""Preserve the pre-update libraries and relocate only the local replay alias."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent
A07=HERE.parent
ROOT=A07.parents[3]
DOC=ROOT/'docs/1M'
SNAP=HERE/'frozen_checkout/ModelLakeFishing'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),(src,dst)
    else:shutil.copy2(src,dst)
files=json.loads((A07.parent/'A0_4/A04_PREFLIGHT.json').read_text())['source_sha256']
for name,expected in files.items():
    source=ROOT/name;assert sha(source)==expected,name
    copy(source,SNAP/name)
for name in ['EVIDENCE_SOURCE_LIBRARY_en.md','EVIDENCE_SOURCE_LIBRARY_zh.md']:
    copy(DOC/name,HERE/'frozen_libraries'/name)
    copy(DOC/name,SNAP/'docs/1M'/name)
for p in (A07/'results').iterdir():
    if p.is_file():copy(p,HERE/'before_library_update_results'/p.name)
old_en='6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd'
assert sha(SNAP/'docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md')==old_en
alias=Path('D:/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing')
assert alias.is_junction() and alias.resolve() in [ROOT.resolve(),SNAP.resolve()]
if alias.resolve()!=SNAP.resolve():
    # Delete only this verified directory junction, without recursive traversal.
    command="[System.IO.Directory]::Delete('"+str(alias)+"', $false); New-Item -ItemType Junction -Path '"+str(alias)+"' -Target '"+str(SNAP)+"' | Out-Null"
    subprocess.run(['powershell','-NoProfile','-Command',command],check=True)
assert alias.resolve()==SNAP.resolve()
records=[]
for name in [*files,'docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md']:
    records.append({'original_resolved_path':str(ROOT/name),'expected_sha256':sha(SNAP/name),
                    'immutable_path':str(SNAP/name),'verified_sha256':sha(SNAP/name)})
proof={'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'Append-only replay-location mapping. Original A0 protocol, raw manifest, measurements and report bytes preserved.',
       'alias':str(alias),'previous_target':str(ROOT),'new_target':str(SNAP),
       'mapping_rule':'Use an immutable path only for the exact original path + expected SHA256 pair.',
       'records':records,
       'preserved_library_hashes':{p.name:sha(p) for p in (HERE/'frozen_libraries').iterdir()},
       'replay':'Run active scale1m.recompute_a0 with the original A0_6/inputs/A0_PROTOCOL.linux.json; authority resolves through this alias to its frozen original bytes. Source-count implementation remains bound to the active unchanged code.'}
(HERE/'REPLAY_PATH_MAP.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'preserved_source_files':len(files),'old_en_sha256':old_en,'alias_target':str(alias.resolve())}))
