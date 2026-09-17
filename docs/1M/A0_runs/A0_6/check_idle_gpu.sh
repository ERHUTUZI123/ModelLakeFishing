set -eu
date -Is
squeue --start -j 1540930
python3 - <<'PY'
import subprocess,re,json
lines=subprocess.run(['scontrol','show','nodes','-o'],capture_output=True,text=True,check=True).stdout.splitlines()
for line in lines:
    d=dict(re.findall(r'(\S+?)=(.*?)(?= \S+?=|$)',line))
    def num(k):
        try:return int(d.get(k,'0'))
        except ValueError:return 0
    def gpu(k):
        m=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',d.get(k,''));return int(m[1]) if m else 0
    free=gpu('CfgTRES')-gpu('AllocTRES')
    if free>0:
        print(json.dumps({'node':d['NodeName'],'state':d['State'],'unallocated_gpu':free,'unallocated_cpu':num('CPUTot')-num('CPUAlloc'),'unallocated_memory_GiB':round((num('RealMemory')-num('AllocMem'))/1024,1),'reason':d.get('Reason','')}))
PY
