"""Re-run independent reporting after its unit-contract correction."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time,hashlib
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];OUT=BASE/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
previous=read(BASE/'library_revision/before_library_update_results/A0_REPORT.json')
s=read(BASE/'STATUS.json');s.update(status='running',phase='independent_report_unit_correction',updated_at_utc=datetime.now(timezone.utc).isoformat());write(BASE/'STATUS.json',s)
cmd=[sys.executable,'-X','utf8','-u','-B','-m','scale1m.recompute_a0','--raw','D:/research/model_lake/data/data1m/a0_20260912/metrics',
     '--protocol',str(BASE.parent/'A0_6/inputs/A0_PROTOCOL.linux.json'),'--inventory',str(BASE/'frozen/A0_METRIC_INVENTORY.json'),
     '--out',str(OUT),'--records',str(OUT/'A0_RECORD_SOURCES.json'),'--source-counts',str(OUT/'A0_SOURCE_COUNTS.json')]
env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
t0=time.perf_counter_ns();started=datetime.now(timezone.utc).isoformat()
with (BASE/'recompute_unit_corrected.log').open('wb') as log:
    p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
t1=time.perf_counter_ns()
record={'command':cmd,'exit_code':p.returncode,'start_ns':t0,'end_ns':t1,'seconds':(t1-t0)/1e9,'started_at_utc':started,
        'reader_sha256':sha(ROOT/'scale1m/recompute_a0.py'),'scope':'Independent report re-read and unit-contract repair; formal training, embeddings, raw query outputs and measured costs reused by verified hash.'}
write(BASE/'library_revision/REPORT_REGENERATION.json',record)
assert p.returncode in [0,2],p.returncode
current=read(OUT/'A0_REPORT.json')
oldvals=previous['new_measurements'];newvals=current['new_measurements']
assert set(oldvals)==set(newvals)
changed=[k for k in oldvals if oldvals[k]['value']!=newvals[k]['value'] or oldvals[k]['status']!=newvals[k]['status']]
assert not changed,changed
assert previous['native_recomputed']==current['native_recomputed']
record.update(raw_measurement_values_and_statuses_unchanged=True,native_quality_rows_identical=True,measurements_compared=len(oldvals))
write(BASE/'library_revision/REPORT_REGENERATION.json',record)
subprocess.run([sys.executable,'-X','utf8','-B',str(BASE/'finalize_delivery.py')],cwd=ROOT,check=True)
s=read(BASE/'STATUS.json');s['steps']['report_unit_correction']=record
s.update(status='incomplete' if not current['completeness']['complete'] else 'complete',phase='recomputed_and_unit_validated',
         completeness=current['completeness'],updated_at_utc=datetime.now(timezone.utc).isoformat())
write(BASE/'STATUS.json',s)
subprocess.run([sys.executable,'-X','utf8','-B',str(BASE/'backfill.py')],cwd=ROOT,check=True)
print(json.dumps(record),flush=True)
