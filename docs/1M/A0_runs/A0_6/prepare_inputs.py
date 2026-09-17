"""Create recorded transport copies; only paths and relocated-file refs change."""
from pathlib import Path
import json,hashlib,shutil,tarfile
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1]
REMOTE='/u801/x98liu/model_lake/a0_eval_20260914'
REPO='/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
previous=read(DOC/'A0_runs/A0_SOURCE_MANIFEST.json')
assert previous['stage']=='A0.5' and previous['status']=='complete'
assert sha(Path(previous['A0_5']['path']))==previous['A0_5']['sha256']
status=read(DOC/'A0_runs/A0_5/STATUS.json');assert status['local_audit_hash_validation']=='PASS'
frozen=OUT/'frozen';frozen.mkdir()
for rel in ['A0.md','A0.5.md','A0_runs/A0_SOURCE_MANIFEST.json','A0_runs/A0_PROTOCOL.json']:
    shutil.copy2(DOC/rel,frozen/Path(rel).name)
inp=OUT/'inputs';(inp/'audit').mkdir(parents=True)
mapping=[('D:/research/model_lake/codes/ModelLakeFishing/docs/1M/A0_runs/audit',REMOTE+'/inputs/audit'),('D:/research/model_lake/codes/ModelLakeFishing',REPO),('D:/research/model_lake/data','/u801/x98liu/model_lake')]
changes=[]
def relocate(obj,trail=''):
    if isinstance(obj,list):return [relocate(v,trail+'/'+str(i)) for i,v in enumerate(obj)]
    if not isinstance(obj,dict):return obj
    out={}
    for k,v in obj.items():
        if k=='path' and isinstance(v,str):
            norm=v.replace('\\','/')
            for src,dst in mapping:
                if norm==src or norm.startswith(src+'/'):
                    out[k]=dst+norm[len(src):];changes.append({'pointer':trail+'/'+k,'before':v,'after':out[k]});break
            else:out[k]=v
        else:out[k]=relocate(v,trail+'/'+k)
    return out
protocol=read(DOC/'A0_runs/A0_PROTOCOL.json')
for name in ['A0_INPUT_AUDIT.json','A0_QUERY_IDENTITY.jsonl','A0_SNAPSHOT_AUDIT.json','A0_INPUT_REVIEW.json']:
    source=DOC/'A0_runs/audit'/name
    bound=next(r for r in protocol['input_audit_binding'].values() if isinstance(r,dict) and Path(r.get('path','')).name==name)
    assert sha(source)==bound['sha256']
    shutil.copy2(source,frozen/name)
    if name=='A0_INPUT_AUDIT.json':write(inp/'audit'/name,relocate(read(source),'/input_audit'))
    else:shutil.copy2(source,inp/'audit'/name)
mapped=relocate(protocol,'/protocol')
ref=mapped['input_audit_binding']['input_audit'];ref['sha256']=sha(inp/'audit/A0_INPUT_AUDIT.json');ref['bytes']=(inp/'audit/A0_INPUT_AUDIT.json').stat().st_size
assert mapped['required_config_audit']==protocol['required_config_audit']
write(inp/'A0_PROTOCOL.linux.json',mapped)
def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys()
        return sum([differences(a[k],b[k],path+'/'+k) for k in a],[])
    if isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b)
        return sum([differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
    return [] if a==b else [path]
diff=differences(protocol,mapped)
assert all(p.endswith('/path') or p in ['/input_audit_binding/input_audit/sha256','/input_audit_binding/input_audit/bytes'] for p in diff)
audit_diff=differences(read(frozen/'A0_INPUT_AUDIT.json'),read(inp/'audit/A0_INPUT_AUDIT.json'))
assert all(p.endswith('/path') for p in audit_diff)
write(OUT/'PATH_RELOCATION.json',{'status':'PASS','mapping':mapping,'changes':changes,'protocol_differences':diff,'audit_differences':audit_diff,'original_protocol_sha256':sha(frozen/'A0_PROTOCOL.json'),'linux_protocol_sha256':sha(inp/'A0_PROTOCOL.linux.json'),'original_input_audit_sha256':sha(frozen/'A0_INPUT_AUDIT.json'),'linux_input_audit_sha256':sha(inp/'audit/A0_INPUT_AUDIT.json'),'training_and_measurement_fields_unchanged':True})
with tarfile.open(OUT/'inputs.tar.gz','w:gz') as tar:
    tar.add(inp,arcname='inputs');tar.add(OUT/'PATH_RELOCATION.json',arcname='PATH_RELOCATION.json')
print(json.dumps({'status':'PASS','protocol_path_changes':len(diff)-2,'audit_path_changes':len(audit_diff),'bundle_sha256':sha(OUT/'inputs.tar.gz')}))
