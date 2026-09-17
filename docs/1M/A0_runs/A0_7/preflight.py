from pathlib import Path
import hashlib,json,subprocess
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
previous=json.loads((BASE.parent/'A0_4/A04_PREFLIGHT.json').read_text())
checks={name:sha(ROOT/name)==digest for name,digest in previous['source_sha256'].items()}
assert all(checks.values()),[n for n,ok in checks.items() if not ok]
assert sha(ROOT/'docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md')=='6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd'
command=['rg','--files','D:/research/model_lake/data',str(ROOT/'docs')]
scan=subprocess.run(command,capture_output=True,text=True,check=True)
names=[p for p in scan.stdout.splitlines() if any(x in Path(p).name.lower() for x in ['crawl','request','discard','duplicat','event','page'])
       and Path(p).suffix.lower() in ['.json','.jsonl','.log','.parquet','.csv']]
evidence={'code_matches_A04':True,'source_files_checked':len(checks),'source_checks':checks,
          'event_log_search':{'command':command,'name_keywords':['crawl','request','discard','duplicat','event','page'],
                             'matching_files':names,'scope':'Current retained data and project documentation; original discarded event records cannot be inferred from retained records.'}}
(BASE/'PREFLIGHT.json').write_text(json.dumps(evidence,indent=2)+'\n')
counts=json.loads((BASE/'results/A0_SOURCE_COUNTS.json').read_text(encoding='utf-8'))
print(json.dumps({'source_files_checked':len(checks),'source_counts':counts['measurements'],
                  'source_missing':counts['missing'],'merge_output_identity':counts.get('merge_output_identity'),
                  'event_log_matches':names},ensure_ascii=False))
