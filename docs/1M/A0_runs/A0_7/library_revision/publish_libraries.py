"""Publish only the two reviewed drafts and verify the historical replay binding."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
A07=HERE.parent
DOC=A07.parents[1]
ROOT=DOC.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
drafts={lang:HERE/f'EVIDENCE_SOURCE_LIBRARY_{lang}.draft.md' for lang in ['en','zh']}
review=read(HERE/'bilingual_final_review.json')
assert review['status']=='AUTOMATED_CHECKS_CLEAR',review['status']
for lang,p in drafts.items():assert sha(p)==review['inputs'][lang]['sha256'],lang
numeric=read(HERE/'audit_en_numeric.json')
assert numeric['status']=='PASS' and numeric['draft_sha256']==sha(drafts['en'])
assert numeric['report_sha256']==sha(A07/'results/A0_REPORT.json')
bilingual_numeric=read(HERE/'audit_bilingual_numeric.json')
assert bilingual_numeric['status']=='PASS'
for p in drafts.values():assert bilingual_numeric['sha256'][str(p)]==sha(p),str(p)
semantic=read(HERE/'SEMANTIC_REVIEW.json')
assert semantic['status']=='PASS'
for lang in drafts:assert semantic['sha256'][lang]==sha(drafts[lang]),lang
audit=read(A07/'results/A07_VALIDATION.json')
assert audit['retrieval_and_cost_recomputation']=='PASS' and len(audit['unit_contract_checks'])==40
assert audit['report_sha256']==sha(A07/'results/A0_REPORT.json')
completeness=read(A07/'results/A0_REPORT.json')['completeness']
assert completeness['missing']==audit['missing']
path_map=read(HERE/'REPLAY_PATH_MAP.json')
original=path_map['preserved_library_hashes']
for record in path_map['records']:
    assert sha(Path(record['immutable_path']))==record['expected_sha256'],record['immutable_path']
plan_bytes=[p.read_bytes() for p in [DOC/'A0.md',A07/'frozen/A0.md']]
plan_blocks=[b[b.index(b'## 1. '):b.index(b'## 10. ')] for b in plan_bytes]
assert plan_blocks[0]==plan_blocks[1],'A0 frozen scientific plan changed'
for lang,draft in drafts.items():
    target=DOC/f'EVIDENCE_SOURCE_LIBRARY_{lang}.md'
    assert sha(target) in [original[target.name],sha(draft)],'Concurrent active-library modification: '+str(target)
for lang,draft in drafts.items():
    target=DOC/f'EVIDENCE_SOURCE_LIBRARY_{lang}.md'
    tmp=target.with_name(target.name+'.a0-update.tmp')
    tmp.write_bytes(draft.read_bytes());tmp.replace(target)
from scale1m.recompute_a0 import verify_manifest
raw=Path('D:/research/model_lake/data/data1m/a0_20260912/metrics')
manifest,hashes=verify_manifest(raw,A07.parent/'A0_6/inputs/A0_PROTOCOL.linux.json',read(A07/'frozen/A0_METRIC_INVENTORY.json'))
assert manifest['stage']=='complete'
records=[ref(p) for p in drafts.values()]
records += [ref(DOC/f'EVIDENCE_SOURCE_LIBRARY_{lang}.md') for lang in drafts]
records += [ref(p) for p in [HERE/'REPLAY_PATH_MAP.json',HERE/'bilingual_final_review.json',HERE/'SEMANTIC_REVIEW.json',
                           HERE/'audit_en_numeric.json',HERE/'audit_bilingual_numeric.json',HERE/'UNIT_CORRECTION.md',HERE/'UNIT_CORRECTION_TESTS.json',
                           HERE/'REPORT_REGENERATION.json',A07/'MANIFEST.json',A07/'results/A0_REPORT.json',
                           A07/'results/A0_METRIC_INVENTORY.json',A07/'results/A07_VALIDATION.json',Path(__file__)]]
for p in [HERE/'audit_method.md',HERE/'audit_numeric.md',HERE/'audit_provenance.md',HERE/'audit_backfill.md',
          HERE/'audit_numeric_check.json',HERE/'UNIT_CORRECTION_INVARIANTS.json',HERE/'audit_bilingual_semantics.md',
          HERE/'check_bilingual.py',HERE/'audit_en_numeric.py',HERE/'audit_bilingual_numeric.py',
          ROOT/'scale1m/recompute_a0.py',ROOT/'scale1m/tests/test_recompute_a0.py']:
    if p.exists():records.append(ref(p))
proof={'schema_version':'a0.evidence-library-revision.v1','status':'published_and_verified',
       'updated_at_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'Only A0.1--A0.7 current method/results; final system X4G+D + HNSW top1000 + task prior -> top10.',
       'historical_authority_sha256':original['EVIDENCE_SOURCE_LIBRARY_en.md'],
       'historical_replay_after_publication':{'status':'PASS','verified_file_paths':len(hashes),'binding_sha256':manifest['binding_sha256']},
       'immutable_replay_records_verified':len(path_map['records']),
       'a0_sections_1_to_9_unchanged_sha256':hashlib.sha256(plan_blocks[0]).hexdigest(),
       'a0_full_inventory_complete':completeness['complete'],'missing_source_events':audit['missing'],
       'artifacts':records}
(HERE/'REVISION_MANIFEST.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':proof['status'],'historical_replay':proof['historical_replay_after_publication'],
                  'libraries':{lang:sha(DOC/f'EVIDENCE_SOURCE_LIBRARY_{lang}.md') for lang in drafts}}),flush=True)
