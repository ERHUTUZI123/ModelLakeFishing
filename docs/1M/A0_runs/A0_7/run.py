"""Execute the frozen independent A0.7 reader on the local i7."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys,time,traceback
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
PROTOCOL=BASE.parent/'A0_6/inputs/A0_PROTOCOL.linux.json'
RAW=Path('D:/research/model_lake/data/data1m/a0_20260912/metrics')
OUT=BASE/'results'
OUT.mkdir(exist_ok=True)
state={'status':'running','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
       'execution_host':'local i7-14650HX','completed_downloads':[],'steps':{}}
env=os.environ.copy()
env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def update(phase,**kw):
    state.update(phase=phase,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kw)
    p=BASE/'STATUS.json.tmp'
    p.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');p.replace(BASE/'STATUS.json')
    print(json.dumps({'phase':phase,**kw}),flush=True)

def start(name,args):
    log=(BASE/(name+'.log')).open('wb')
    cmd=[sys.executable,'-X','utf8','-u','-B']+args
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['steps'][name]={'command':cmd,'pid':p.pid,'started_at_utc':datetime.now(timezone.utc).isoformat(),
                          'start_monotonic':time.monotonic()}
    update(name+'_started')
    return p,log

def finish(name,pair,allowed=(0,)):
    p,log=pair
    while p.poll() is None:
        time.sleep(10)
        update(name+'_running')
    log.close()
    s=state['steps'][name]
    s.update(exit_code=p.returncode,elapsed_seconds=time.monotonic()-s['start_monotonic'])
    update(name+'_finished')
    assert p.returncode in allowed,(name,p.returncode,'see '+str(BASE/(name+'.log')))

def download(rec):
    path=Path(rec['local']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256'],path
        return str(path)
    part=path.with_name(path.name+'.part')
    for attempt in range(3):
        with (BASE/('download_'+path.parent.name+'_'+path.name+'.log')).open('ab') as log:
            p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',
                'x98liu@watgpu.cs.uwaterloo.ca:'+rec['remote'],str(part)],stdout=log,stderr=log)
        if p.returncode==0:break
        if attempt==2:raise RuntimeError('Download failed: '+rec['remote'])
        time.sleep(5)
    assert part.stat().st_size==rec['bytes'] and sha(part)==rec['sha256'],path
    part.replace(path)
    return str(path)

def main():
    update('starting',implementation_sha256={n:sha(ROOT/'scale1m'/n) for n in
        ['recompute_a0.py','a0_recompute_checks.py','a0_source_counts.py']})
    source=start('source_counts',['-m','scale1m.a0_source_counts','--protocol',str(PROTOCOL),
        '--historical-bindings',str(BASE.parent/'A0_2/A0_HISTORICAL_SOURCE_IDENTITY.json'),
        '--out',str(OUT/'A0_SOURCE_COUNTS.json'),'--run-id','A0_20260912'])
    plan=json.loads((BASE/'DOWNLOAD_PLAN.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(download,r) for r in plan['files']]
        for future in as_completed(futures):
            state['completed_downloads'].append(future.result())
            update('downloading',total_downloads=len(futures))
    finish('source_counts',source,(0,2))
    common=['-m','scale1m.recompute_a0','--raw',str(RAW),'--protocol',str(PROTOCOL),
            '--inventory',str(BASE/'frozen/A0_METRIC_INVENTORY.json'),'--out',str(OUT)]
    finish('record_template',start('record_template',common+['--make-record-template',str(OUT/'A0_RECORD_SOURCES.json')]))
    finish('recompute',start('recompute',common+['--records',str(OUT/'A0_RECORD_SOURCES.json'),
        '--source-counts',str(OUT/'A0_SOURCE_COUNTS.json')]),(0,2))
    report=json.loads((OUT/'A0_REPORT.json').read_text(encoding='utf-8'))
    update('recomputed',status='complete' if report['completeness']['complete'] else 'incomplete',
           completeness=report['completeness'],all_ann_fidelity_passed=report['all_ann_fidelity_passed'])

if __name__=='__main__':
    try:main()
    except Exception:
        update('failed',status='failed',error=traceback.format_exc())
        raise
