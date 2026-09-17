"""A0.3 recorded commands, resource samples, and frozen A0.2 preflight."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUN=Path('D:/research/model_lake/runs/A0_20260912/smoke_s0_e1')

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['entry','boundary','regression','smoke']);p.add_argument('--attempt',type=int,default=1);args=p.parse_args()
    if args.stage=='entry':
        target=OUT/'A0_3_ENTRY.json';assert not target.exists()
        manifest=ROOT/'docs/1M/A0_runs/A0_2/A0_2_MANIFEST.json'
        validated=json.loads((manifest.parent/'A0_DELIVERY_VALIDATION.json').read_text(encoding='utf-8'))
        assert sha(manifest)==validated['A0_2_manifest']['sha256']
        capture=json.loads((manifest.parent/'A0_CODE_CAPTURE.json').read_text(encoding='utf-8'))
        for ref in capture['sources']: assert sha(ref['path'])==ref['sha256'],ref['path']
        frozen=OUT/'frozen';frozen.mkdir(exist_ok=False)
        for rel in ['A0.md','A0.2.md','A0_runs/A0_SOURCE_MANIFEST.json','A0_runs/A0_PROTOCOL.json','A0_runs/A0_COMMANDS.md','A0_runs/A0_DISCREPANCIES.md']:
            shutil.copy2(ROOT/'docs/1M'/rel,frozen/Path(rel).name)
        write(target,{'stage':'A0.3','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'A02_manifest_sha256':sha(manifest),'A02_source_files_verified':len(capture['sources']),
            'ram':psutil.virtual_memory()._asdict(),'disk':psutil.disk_usage('D:/')._asdict(),
            'scope':'A0.3 only; no formal training','smoke_dir':str(RUN)})
        print('A0.2 source preflight PASS');return 0
    command=[sys.executable,'-X','utf8','-B','-m']
    if args.stage=='boundary': command+=['pytest','scale1m/tests/test_a03_information_boundary.py','-q']
    elif args.stage=='regression':command+=['pytest','scale1m/tests','stage2TrainGraphSAGE/tests','-q']
    else:
        assert not RUN.exists(),'Preserve previous smoke output'
        command+=['scale1m.train_rung','--rung','full','--graph','D:/research/model_lake/data/data1m/a0_20260912/graph',
            '--out',str(RUN),'--seed','0','--epochs','1','--family-vocab','D:/research/model_lake/data/data1m/feats_rf/family_vocab.csv',
            '--fanout','--sparse-M','--contrast-n-neg','256','--chunked-infer','50000','--skip-diagnostics',
            '--lake-gamma','0.5','--global-n-datasets','128','--smoke-only']
    label=args.stage if args.attempt==1 else args.stage+'_'+str(args.attempt)
    log=OUT/(label+'.log');record=OUT/(label+'.json')
    assert not record.exists() and not log.exists(),'Preserve previous execution evidence'
    if args.stage!='smoke':command+=['--junitxml='+str(OUT/(label+'.xml'))]
    start=time.perf_counter();peak=0;samples=[]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    with log.open('wb') as f:
        child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                mem=process.memory_info();peak=max(peak,getattr(mem,'peak_wset',mem.rss))
                samples.append({'seconds':time.perf_counter()-start,'rss':mem.rss,'available_ram':psutil.virtual_memory().available})
            except psutil.Error:pass
            time.sleep(2)
    write(record,{'stage':'A0.3','operation':args.stage,'argv':command,'cwd':str(ROOT),'pid':child.pid,
        'returncode':child.returncode,'elapsed_seconds':time.perf_counter()-start,'peak_process_working_set_bytes':peak,
        'samples':samples,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'log_sha256':sha(log)})
    print(json.dumps({'operation':args.stage,'returncode':child.returncode,'seconds':time.perf_counter()-start,'log':str(log)}))
    return child.returncode

if __name__=='__main__':raise SystemExit(main())
