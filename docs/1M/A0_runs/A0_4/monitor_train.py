"""Observe the training process tree, OS RSS high-water mark and GPU samples."""
import json,os,resource,subprocess,sys,time,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE/'ModelLakeFishing'
SEED=int(sys.argv[1]);assert SEED in (0,1,2)
RUN=Path(f'/u801/x98liu/model_lake/runs/A0_20260912/A0GD_full_s{SEED}_e25')
OUT=BASE/f'observations_s{SEED}';OUT.mkdir(exist_ok=False)
assert not RUN.exists(),'Fresh formal output required'
command=[sys.executable,'-u','-m','scale1m.train_rung','--rung','full','--graph','/u801/x98liu/model_lake/data1m/a0_20260912/graph','--out',str(RUN),'--seed',str(SEED),'--epochs','25','--family-vocab','/u801/x98liu/model_lake/data1m/feats_rf/family_vocab.csv','--fanout','--sparse-M','--contrast-n-neg','256','--chunked-infer','50000','--skip-diagnostics','--lake-gamma','0.5','--global-n-datasets','128']
(OUT/'command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic();peak=0;gpupeak=0;gpuvalid=False
with (OUT/'train.log').open('w') as log,(OUT/'resources.jsonl').open('w') as samples:
    p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
        entries={}
        for status in Path('/proc').glob('[0-9]*/status'):
            try:
                fields=dict(line.split(':',1) for line in status.read_text().splitlines() if ':' in line)
                entries[int(status.parent.name)]=(int(fields['PPid']),int(fields.get('VmRSS','0 kB').split()[0])*1024)
            except (OSError,ValueError,KeyError):pass
        ids={p.pid}
        while True:
            new=ids|{pid for pid,(parent,rss) in entries.items() if parent in ids}
            if new==ids:break
            ids=new
        rss=sum(entries.get(pid,(0,0))[1] for pid in ids);peak=max(peak,rss)
        try:
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        except (OSError,subprocess.TimeoutExpired):
            gpu=subprocess.CompletedProcess([],1,'','GPU sample unavailable')
        gm=None
        if gpu.returncode==0:
            try:
                gm=sum(int(line.split(',')[1].strip()) for line in gpu.stdout.splitlines() if int(line.split(',')[0].strip()) in ids)
                gpupeak=max(gpupeak,gm);gpuvalid=True
            except ValueError:pass
        samples.write(json.dumps({'seconds':time.monotonic()-start,'process_tree_pids':sorted(ids),'rss_sum_bytes':rss,'gpu_process_memory_mib':gm})+'\n');samples.flush()
        time.sleep(2)
record={'returncode':p.returncode,'elapsed_seconds':time.monotonic()-start,'sampled_peak_process_tree_rss_bytes':peak,'os_child_maxrss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'sampled_peak_gpu_process_memory_mib':gpupeak if gpuvalid else None,'sampling_seconds':2,'rss_scope':'Linux process-tree sampled sum plus OS direct-child high-water mark','gpu_scope':'nvidia-smi sampled process memory; exact PyTorch allocation peak in A0_RUN_RECORDS.json','run':str(RUN)}
(OUT/'RESOURCE_REPORT.json').write_text(json.dumps(record,indent=2)+'\n')
if (RUN/'stdout').is_dir():shutil.copy2(OUT/'train.log',RUN/'stdout/train.log')
print(json.dumps(record),flush=True)
if p.returncode==0:
    import torch
    report=json.loads((RUN/'MANIFEST.json').read_text())
    native=json.loads((RUN/'A0_RUN_RECORDS.json').read_text())
    expected=json.loads((ROOT/'docs/1M/A0_runs/A0_PROTOCOL.json').read_text())['required_config_audit']['resolved_config']
    actual=json.loads((RUN/'metadata/resolved_config.json').read_text())['resolved_config']
    assert actual==expected and len(actual)==40
    assert report['epochs']==25 and report['start_epoch']==0 and report['seed']==SEED and report['resumed_from'] is None
    assert report['mechanism_gate']['passed'] and native['status']=='complete' and native['timing_complete']
    assert len(native['history'])==25
    assert report['graph_sha256']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
    ckpt=RUN/'ckpt/last.pt'
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert ck['epoch']==24 and len(ck['history'])==25
    assert not ck.get('extra',{}).get('smoke_only')
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    assert sha(ckpt)==native['a0']['checkpoint_sha256']
    proof={'status':'PASS','seed':SEED,'epochs':25,'checkpoint_epoch':24,'checkpoint_sha256':sha(ckpt),'config_keys_equal':40,'graph_sha256':report['graph_sha256'],'loss_first':native['history'][0]['total'],'loss_last':native['history'][-1]['total'],'mechanism_gate':report['mechanism_gate'],'resource_report':record,'run':str(RUN)}
    (OUT/'A04_VALIDATION.json').write_text(json.dumps(proof,indent=2)+'\n')
raise SystemExit(p.returncode)
