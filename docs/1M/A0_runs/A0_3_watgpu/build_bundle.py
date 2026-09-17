import hashlib,json,tarfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
GRAPH=Path('D:/research/model_lake/data/data1m/a0_20260912/graph')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
capture=json.loads((ROOT/'docs/1M/A0_runs/A0_2/A0_CODE_CAPTURE.json').read_text(encoding='utf-8'))
for x in capture['sources']:assert sha(Path(x['path']))==x['sha256'],x['path']
files={Path(x['path']).relative_to(ROOT):Path(x['path']) for x in capture['sources']}
for folder in ['scale','scale1m','stage2TrainGraphSAGE','stage3HNSW','stage1BuildTransferGraph','repro']:
    for p in (ROOT/folder).rglob('*.py'):files[p.relative_to(ROOT)]=p
for name in ['A0_PROTOCOL.json']:
    files[Path('docs/1M/A0_runs')/name]=ROOT/'docs/1M/A0_runs'/name
files[Path('docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md')]=ROOT/'docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md'
for name in ['monitor_smoke.py','smoke.sbatch']:files[Path('a0_ops')/name]=OUT/name
manifest={'sources':{str(k).replace('\\','/'):sha(p) for k,p in files.items()},'A02_source_files_verified':len(capture['sources']),
 'family_vocab_sha256':sha(Path('D:/research/model_lake/data/data1m/feats_rf/family_vocab.csv'))}
(OUT/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
with tarfile.open(OUT/'a03_smoke_bundle.tar.gz','w:gz') as tar:
    for k,p in files.items():tar.add(p,arcname='ModelLakeFishing/'+k.as_posix())
    tar.add(OUT/'BUNDLE_MANIFEST.json',arcname='BUNDLE_MANIFEST.json')
    for name in ['x_dataset.npy','meta.json','A0_FEATURE_REPAIR.json']:tar.add(GRAPH/name,arcname='graph_patch/'+name)
    tar.add(OUT/'prepare_remote.py',arcname='prepare_remote.py')
print(json.dumps({'sources':len(files),'bundle_bytes':(OUT/'a03_smoke_bundle.tar.gz').stat().st_size,'bundle_sha256':sha(OUT/'a03_smoke_bundle.tar.gz')}))
