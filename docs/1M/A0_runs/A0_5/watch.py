"""Monitor A0.5 and collect verified audit artifacts; never launches new jobs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess,time,tarfile,traceback
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1]
HOST='x98liu@watgpu.cs.uwaterloo.ca'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)
SCRIPT='''python3 - <<'PY'
import json,subprocess,hashlib,tarfile
from pathlib import Path
base=Path('/u801/x98liu/model_lake/a0_export_20260913')
job=base.joinpath('job_id.txt').read_text().strip()
def cmd(args):return subprocess.run(args,capture_output=True,text=True).stdout
acct=cmd(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,MaxRSS','-n','-P'])
queue=cmd(['squeue','-j',job,'-h','-o','%i|%T|%R'])
start=cmd(['squeue','--start','-j',job])
r={'job_id':job,'state':'UNKNOWN','sacct':acct,'queue':queue,'start_estimate':start,'seeds':[]}
for line in acct.splitlines():
    parts=line.split('|')
    if parts[0]==job:r.update(state=parts[1],exit_code=parts[2],elapsed=parts[3])
for line in queue.splitlines():
    parts=line.split('|')
    if parts[0]==job:r.update(state=parts[1],reason=parts[2])
for seed in range(3):
    row={'seed':seed,'phase':'waiting'}
    for name in ['export','prior','validation']:
        log=base/f'{name}_s{seed}.log'
        if log.exists():row.update(phase=name,tail=log.read_text(errors='replace').splitlines()[-8:])
    proof=base/f'A05_VALIDATION_s{seed}.json'
    if proof.exists():row.update(phase='PASS',validation=json.loads(proof.read_text()))
    r['seeds'].append(row)
log=base/f'slurm-{job}.log'
if log.exists():r['slurm_tail']=log.read_text(errors='replace').splitlines()[-10:]
if r['state']=='COMPLETED' and all(s['phase']=='PASS' for s in r['seeds']):
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    bundle=base/'audit_delivery.tar.gz'
    if not bundle.exists():
        with tarfile.open(bundle,'w:gz') as tar:
            for p in base.iterdir():
                if p.is_file() and p.suffix in ['.json','.log','.time','.sbatch','.txt','.py']:tar.add(p,arcname='operations/'+p.name)
            for row in r['seeds']:
                seed=row['seed'];ex=base.parent/f'data1m/a0_20260912/exports/A0GD_full_s{seed}_e25'
                for name,v in row['validation']['files'].items():
                    p=ex/name;assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],name
                    if name not in ['z_m.npy','z_m_eval.npy','z_d.npy','z_d_eval.npy','model_ids.parquet']:tar.add(p,arcname=f'export_s{seed}/'+name)
    r['audit_bundle']={'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size}
print(json.dumps(r))
PY
'''
def refresh():
    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'bash','-s'],input=SCRIPT.encode(),capture_output=True,timeout=150)
    if process.returncode:raise RuntimeError(process.stdout.decode(errors='replace')+process.stderr.decode(errors='replace'))
    s=json.loads(process.stdout);s['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    if 'audit_bundle' in s:
        dest=OUT/'audit_delivery.tar.gz'
        if not dest.exists():
            p=subprocess.run(['scp','-q','-o','BatchMode=yes',HOST+':'+s['audit_bundle']['path'],str(dest)],capture_output=True,timeout=150)
            assert p.returncode==0,p.stderr
        assert sha(dest)==s['audit_bundle']['sha256']
        delivery=OUT/'delivery'
        if not delivery.exists():
            delivery.mkdir()
            with tarfile.open(dest) as tar:tar.extractall(delivery,filter='data')
        for row in s['seeds']:
            folder=delivery/f"export_s{row['seed']}"
            for p in folder.iterdir():
                expected=row['validation']['files'][p.name]
                assert sha(p)==expected['sha256'] and p.stat().st_size==expected['bytes']
        s['local_audit_hash_validation']='PASS'
    fail=s['state'].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'] or (s['state']=='COMPLETED' and 'audit_bundle' not in s)
    s['stage_status']='complete' if s.get('local_audit_hash_validation')=='PASS' else ('blocked' if fail else 'running')
    write(OUT/'STATUS.json',s)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':s['updated_at_utc'],'state':s['state'],'reason':s.get('reason'),'phases':[r['phase'] for r in s['seeds']]})+'\n')
    backfill(s)
    print(json.dumps({'status':s['stage_status'],'job':s['job_id'],'state':s['state'],'reason':s.get('reason'),'phases':[r['phase'] for r in s['seeds']]}),flush=True)
    return s['stage_status'] in ['complete','blocked']
def backfill(s):
    label={'complete':'已完成','blocked':'受阻','running':'进行中'}[s['stage_status']]
    rows=[]
    for row in s['seeds']:
        v=row.get('validation',{})
        rows.append(f"| {row['seed']} | {row['phase']} | {v.get('queries','待校验')} | {v.get('visible_edges','待校验')} | {v.get('tasks','待校验')} | {v.get('export_seconds','待测量')} |")
    text=f'''# A0.5 执行回填

状态：**{label}**。更新于 {s['updated_at_utc']}。当前作业 **{s['job_id']}**：`{s['state']}`，原因/节点：`{s.get('reason','见最终记录')}`。

## 1. 逐 seed 结果

| seed | 当前步骤/验收 | 有效 gold 查询 | prior 可见边 | task 数 | embed 秒 |
|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

阶段步骤为 waiting → export → prior → validation → PASS。预期有效查询为 1476/1101/1545；prior 可见边为 198216/196912/196124；实际值通过校验后填入。

## 2. 执行及输入绑定

A0.4 三个 25 epoch 末 checkpoint 已验收，本轮固定使用各 run 的 `ckpt/last.pt`（epoch 24）。本地及远端 checkpoint SHA、训练源码、ladder、task-nodes 均已核验。来源见 [LOCAL_PREFLIGHT.json](A0_runs/A0_5/LOCAL_PREFLIGHT.json) 和 [远端输入核验](A0_runs/A0_5/preflight.sh.log)。

按 [A0.md](A0.md) 依次对 seed 0、1、2 执行：

```text
python -m scale1m.export_rf --run RUN --stage embed --ckpt last --graph GRAPH --ladder LADDER --chunk 50000 --out EXPORT
python -m stage3HNSW.build_prior_sidecar --graph-store GRAPH --export EXPORT --split-seed SEED --task-nodes TASK_NODES
```

实际绝对路径与原参数见 [run.sbatch](A0_runs/A0_5/run.sbatch)。作业 1540850 曾解除单节点限制，后在 watgpu908 环境准备阶段因缺 pip 失败。当前改投此前成功使用过的 watgpu608，使用 `gpu:1`；保留 8 CPU、128 GiB RAM、1 小时时限，增加 Python 3.11 环境兼容性预检。实际重投命令见 [retry_608.sh](A0_runs/A0_5/retry_608.sh)，计算参数一致。

- 源码：`/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing`。
- 新图：`/u801/x98liu/model_lake/data1m/a0_20260912/graph`，digest `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`。
- 输出：`/u801/x98liu/model_lake/data1m/a0_20260912/exports/A0GD_full_s{0,1,2}_e25`。
- task-nodes：`/u801/x98liu/model_lake/data1m/rf/canon/dataset_nodes_merged.parquet`，SHA `763201d8e6e103a664b658ec7148b657d28f1d0cdc6f70a54b06687122dbdc57`。

## 3. 实际验收项目

每个 seed 完成后，独立验收脚本检查：四份新向量的 128 维、float32、全量有限值、L2 范数误差 <1e-5；原 exporter 的分块/完整推理误差 <1e-5；模型和 dataset 行表与图逐行一致。

新 gold 候选使用原导出器从图和 split 重建；有效查询 ID 顺序、root、全部候选 ID 和 oriented values 的哈希逐条比对 A0.1 的 4122 条冻结记录。此验证也绑定 gold 模型身份。

prior sidecar 按原协议保存可见记录和 root/task 映射。额外调用实际 TaskPrior 消费者重新聚合，并用独立逐记录 sum/count 验证 `(sum + 0.5×5)/(count+5)`；逐 task 检查无证据模型返回 0。确认可见边顺序与 A0.1 一致、scored test roots 全部排除、task 字符串映射正确且不退化。

产物 SHA 和字节数记录于每个 `A05_VALIDATION_s{seed}.json`；完整向量保存在 watgpu，上述远端目录供 A0.6 直接使用。三个 seed 全部通过后，审核报告、日志、sidecar、gold 和 dataset 行表会回收到本地 `A0_runs/A0_5/delivery` 并再次核验哈希。

## 4. 反馈区

| 问题 | 实际处理 / 当前反馈 |
|---|---|
| 远端缺 task-nodes | 从本地冻结文件上传，SHA 与 A0.1 一致，输入预检通过 |
| 原作业 1540848 等待 watgpu308 | 节点 DRAIN，管理员原因 Kill task failed；作业在 PENDING 阶段取消，保留记录 |
| 作业 1540850 | watgpu908 已分配，环境阶段 No module named pip 失败；三个 export 目录仍不存在 |
| 新作业 1540878 | 改投 watgpu608；查询时有 3 张未分配 GPU、17 CPU、约 198 GiB 未分配主存；增加 Python 3.11 预检，调度状态见本页顶部 |
| SSH 偶发连接错误 | 重试已恢复；后台监测遇到错误保存 watch_errors.log 并继续查询 |
| 自动回填 | 每约 45 秒查询；逐 seed PASS 及最终审核文件本地哈希通过后才标完成 |
| 最终系统 | X4G+D → HNSW top1000 → task prior → top10，A0.6 负责后续检索评测 |
| 既有证据缺口 | 两个上游事件计数保持 missing，A0.7 继续处理 |

最新调度预估（服务器时间，可能变化）：

```text
'''+s.get('start_estimate','')+'''```

证据：[STATUS.json](A0_runs/A0_5/STATUS.json)、[events.jsonl](A0_runs/A0_5/events.jsonl)、[校验脚本](A0_runs/A0_5/validate.py)。
'''
    (DOC/'A0.5.md').write_text(text,encoding='utf-8')
    p=DOC/'A0.md';lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 当前阶段 |'):lines[i]=f'| 当前阶段 | A0.1–A0.4 已完成；A0.5 {label}；A0.6–A0.7 待执行 |'
        elif line.startswith('| 最后更新 |'):lines[i]=f"| 最后更新 | {s['updated_at_utc']}；A0.5 {label} |"
        elif line.startswith('| 当前可做的下一步 |'):lines[i]='| 当前可做的下一步 | '+('A0.6 检索评测' if s['stage_status']=='complete' else '等待/监测 A0.5 作业，逐 seed 校验新向量与 prior')+' |'
        elif line.startswith('| 本次已完成 |'):lines[i]='| 本次已完成 | A0.5 前置输入核验及提交，逐 seed 实测结果见 [A0.5.md](A0.5.md) |'
        elif line.startswith('| A0.5 导出与 prior |'):lines[i]=f"| A0.5 导出与 prior | {label} | 2026-09-13 EDT 起 | [A0.5.md](A0.5.md)、[A0_5](A0_runs/A0_5/) | 作业 {s['job_id']}，逐 seed 验收见回填 | 三 seed export/prior 验收 |"
        for row in s['seeds']:
            if line.startswith(f"| {row['seed']} | 1539537_"):
                fields=line.split('|');fields[4]=' '+('PASS' if row['phase']=='PASS' else row['phase'])+' ';lines[i]='|'.join(fields)
    current='\n'.join(lines)+'\n'
    old=(OUT/'frozen/A0.md').read_text(encoding='utf-8')
    assert current[current.index('## 1.'):current.index('## 10.')]==old[old.index('## 1.'):old.index('## 10.')]
    p.write_text(current,encoding='utf-8')
    previous=read(OUT/'frozen/A0_SOURCE_MANIFEST.json')
    items=[DOC/'A0.5.md',DOC/'A0.md',OUT/'STATUS.json',OUT/'events.jsonl',OUT/'validate.py',OUT/'run.sbatch',OUT/'reschedule.sh',OUT/'LOCAL_PREFLIGHT.json',OUT/'watch.py']
    if (OUT/'audit_delivery.tar.gz').exists():items.append(OUT/'audit_delivery.tar.gz')
    write(OUT/'MANIFEST.json',{'stage':'A0.5','status':s['stage_status'],'updated_at_utc':s['updated_at_utc'],'artifacts':[ref(p) for p in items],'previous_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json')})
    write(DOC/'A0_runs/A0_SOURCE_MANIFEST.json',{'schema':'a0.source_manifest.v1','stage':'A0.5','status':s['stage_status'],'authority':previous['authority'],'A0_2':previous['A0_2'],'A0_3':previous['A0_3'],'A0_4':previous['A0_4'],'A0_5':ref(OUT/'MANIFEST.json'),'previous_stage_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json'),'later_stages':'A0.6-A0.7 pending'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    for _ in range(1920):
        try:
            if refresh() or args.once:return
        except Exception:
            with (OUT/'watch_errors.log').open('a',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n'+traceback.format_exc())
            if args.once:raise
        time.sleep(45)
if __name__=='__main__':main()
