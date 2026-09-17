set -eu
date -Is
squeue -u x98liu -o '%i %j %T %R'
sacct -j 1540850 --format=JobID,State,ExitCode,Elapsed,NodeList -n -P
sinfo -N -o '%N %t %m %G' | sort -u
python3 - <<'PY'
import subprocess,re,json
result=subprocess.run(['scontrol','show','nodes','-o'],capture_output=True,text=True,check=True)
for line in result.stdout.splitlines():
    fields=dict(re.findall(r'(\S+?)=(.*?)(?= \S+?=|$)',line))
    def number(key):
        try:return int(fields.get(key,'0'))
        except ValueError:return 0
    def gpu(tres):
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        return int(match.group(1)) if match else 0
    total=gpu(fields.get('CfgTRES','')); used=gpu(fields.get('AllocTRES',''))
    print(json.dumps({'node':fields.get('NodeName'),'state':fields.get('State'),'gpu_total':total,'gpu_allocated':used,'gpu_unallocated':total-used,'cpu_unallocated':number('CPUTot')-number('CPUAlloc'),'memory_unallocated_MiB':number('RealMemory')-number('AllocMem'),'memory_free_observed_MiB':number('FreeMem'),'gres':fields.get('Gres'),'partitions':fields.get('Partitions'),'reason':fields.get('Reason','')}))
PY
sacctmgr -n -P show assoc where user=x98liu format=Account,Partition,QOS,DefaultQOS
