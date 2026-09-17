import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=Path('/u801/x98liu/model_lake/data1m')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
for rel,expected in manifest['sources'].items():assert sha(ROOT/'ModelLakeFishing'/rel)==expected,rel
assert sha(DATA/'feats_rf/family_vocab.csv')==manifest['family_vocab_sha256']
patch=ROOT/'graph_patch'
meta=json.loads((patch/'meta.json').read_text())
dest=DATA/'a0_20260912/graph';assert not dest.exists(),dest
dest.parent.mkdir(parents=True,exist_ok=True)
tmp=dest.parent/'graph_transfer_pending';tmp.mkdir()
for name,expected in meta['files'].items():
    src=patch/name if (patch/name).exists() else DATA/'graphs/hgraph_rf'/name
    assert sha(src)==expected,name
    shutil.copy2(src,tmp/name)
    assert sha(tmp/name)==expected,name
shutil.copy2(patch/'meta.json',tmp/'meta.json')
assert sha(tmp/'meta.json')==sha(patch/'meta.json')
tmp.rename(dest)
(ROOT/'REMOTE_TRANSFER_VALIDATION.json').write_text(json.dumps({'source_files_verified':len(manifest['sources']),'graph_files_verified':len(meta['files']),'graph':str(dest),'family_vocab_sha256':manifest['family_vocab_sha256'],'status':'PASS'},indent=2)+'\n')
print('PASS: isolated sources, vocabulary and exact prepared graph bytes verified',flush=True)
