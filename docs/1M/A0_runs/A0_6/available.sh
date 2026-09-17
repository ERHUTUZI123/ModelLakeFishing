python3 - <<'PY'
import subprocess,re,json
out=subprocess.run(['scontrol','show','nodes','-o'],capture_output=True,text=True,check=True).stdout
for line in out.splitlines():
    d=dict(re.findall(r'(\S+?)=(.*?)(?= \S+?=|$)',line))
    def n(k):
        try:return int(d.get(k,'0'))
        except ValueError:return 0
    def g(k):
        m=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',d.get(k,''));return int(m[1]) if m else 0
    gpu=g('CfgTRES')-g('AllocTRES');cpu=n('CPUTot')-n('CPUAlloc');mem=n('RealMemory')-n('AllocMem')
    if gpu>0 and cpu>=8 and mem>=131072 and not any(x in d.get('State','') for x in ['DRAIN','DOWN','NOT_RESPONDING']):print(json.dumps({'node':d['NodeName'],'gpu_unallocated':gpu,'cpu_unallocated':cpu,'memory_unallocated_MiB':mem,'state':d['State']}))
PY
