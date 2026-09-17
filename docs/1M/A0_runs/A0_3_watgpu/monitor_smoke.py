"""Observe the training process tree, OS RSS high-water mark and GPU samples."""
import json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/u801/x98liu/model_lake/runs/A0_20260912/smoke_s0_e1_retry1')
OUT=ROOT.parent/'observations';OUT.mkdir(exist_ok=False)
assert not RUN.exists(),'Fresh smoke output required'
command=[sys.executable,'-u','-m','scale1m.train_rung','--rung','full','--graph','/u801/x98liu/model_lake/data1m/a0_20260912/graph','--out',str(RUN),'--seed','0','--epochs','1','--family-vocab','/u801/x98liu/model_lake/data1m/feats_rf/family_vocab.csv','--fanout','--sparse-M','--contrast-n-neg','256','--chunked-infer','50000','--skip-diagnostics','--lake-gamma','0.5','--global-n-datasets','128','--smoke-only']
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
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        gm=None
        if gpu.returncode==0:
            try:
                gm=sum(int(line.split(',')[1].strip()) for line in gpu.stdout.splitlines() if int(line.split(',')[0].strip()) in ids)
                gpupeak=max(gpupeak,gm);gpuvalid=True
            except ValueError:pass
        samples.write(json.dumps({'seconds':time.monotonic()-start,'process_tree_pids':sorted(ids),'rss_sum_bytes':rss,'gpu_process_memory_mib':gm})+'\n');samples.flush()
        time.sleep(2)
record={'returncode':p.returncode,'elapsed_seconds':time.monotonic()-start,'sampled_peak_process_tree_rss_bytes':peak,'os_child_maxrss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'sampled_peak_gpu_process_memory_mib':gpupeak if gpuvalid else None,'sampling_seconds':2,'rss_scope':'Linux process-tree sampled sum plus OS direct-child high-water mark','gpu_scope':'nvidia-smi sampled process memory; exact PyTorch allocation peak in SMOKE_REPORT.json','run':str(RUN)}
(OUT/'RESOURCE_REPORT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(p.returncode)
