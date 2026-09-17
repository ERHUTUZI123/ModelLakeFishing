"""Durable A0.4 status, result collection and document backfill; no submissions."""
import argparse,hashlib,json,subprocess,sys,time,shutil,traceback
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1]
REMOTE='/u801/x98liu/model_lake/a0_formal_20260913'
HOST='x98liu@watgpu.cs.uwaterloo.ca'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)
def run(argv,**kw):return subprocess.run(argv,capture_output=True,timeout=90,**kw)
SCRIPT='''python3 - <<'PY'
import json,subprocess,hashlib
from pathlib import Path
base=Path('/u801/x98liu/model_lake/a0_formal_20260913')
sacct=subprocess.run(['sacct','-j','1539537','--format=JobID,State,ExitCode,Elapsed,MaxRSS','-n','-P'],capture_output=True,text=True).stdout
queue=subprocess.run(['squeue','-r','-j','1539537','-h','-o','%i|%T|%R'],capture_output=True,text=True).stdout
result={'job_array':1539537,'sacct':sacct,'queue':queue,'seeds':[]}
for seed in range(3):
    run=base.parent/f'runs/A0_20260912/A0GD_full_s{seed}_e25'
    obs=base/f'observations_s{seed}'
    row={'seed':seed,'run':str(run),'epochs_saved':0,'state':'PENDING','reason':''}
    for line in sacct.splitlines():
        vals=line.split('|')
        if vals[0]==f'1539537_{seed}':row.update(state=vals[1],exit_code=vals[2],elapsed=vals[3],maxrss=vals[4])
    for line in queue.splitlines():
        vals=line.split('|')
        if vals[0]==f'1539537_{seed}':row.update(state=vals[1],reason=vals[2])
    for path,key in [(run/'A0_RUN_RECORDS.json','native'),(obs/'A04_VALIDATION.json','validation'),(obs/'RESOURCE_REPORT.json','resources'),(run/'metadata/resolved_config.json','config')]:
        if path.exists():
            try:row[key]=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):pass
    if 'native' in row:
        history=row['native'].get('history',[]);row['epochs_saved']=len(history)
        if history:row['last_loss']=history[-1].get('total')
    for path,key in [(obs/'train.log','train_tail'),(base/f'slurm-1539537_{seed}.log','slurm_tail')]:
        if path.exists():row[key]=path.read_text(errors='replace').splitlines()[-5:]
    if row['state']=='COMPLETED' and 'validation' in row:
        table={}
        for path in run.rglob('*'):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                table[str(path.relative_to(run))]=h.hexdigest()
        row['run_sha256']=table
    result['seeds'].append(row)
print(json.dumps(result))
PY
'''
def refresh():
    result=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'bash','-s'],input=SCRIPT.encode())
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    state=json.loads(result.stdout);state['checked_at_utc']=datetime.now(timezone.utc).isoformat()
    expected=read(DOC/'A0_runs/A0_PROTOCOL.json')['required_config_audit']['resolved_config']
    for row in state['seeds']:
        if 'config' in row:
            assert row['config']['resolved_config']==expected,'40-key config mismatch'
            assert row['config']['graph_sha256']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
        if row['state']=='COMPLETED' and 'validation' in row:
            target=OUT/f'delivery_s{row["seed"]}'
            if not target.exists():
                tmp=OUT/f'download_s{row["seed"]}';tmp.mkdir(exist_ok=True)
                cmd=run(['scp','-q','-r','-o','BatchMode=yes',HOST+':'+row['run'],str(tmp)])
                assert cmd.returncode==0,cmd.stderr
                fetched=tmp/Path(row['run']).name
                for rel,digest in row['run_sha256'].items():assert sha(fetched/rel)==digest,rel
                fetched.rename(target)
            for rel,digest in row['run_sha256'].items():assert sha(target/rel)==digest,rel
            row['local_delivery']=str(target)
            row['local_hash_validation']='PASS'
    complete=all(r.get('local_hash_validation')=='PASS' and r['validation']['status']=='PASS' for r in state['seeds'])
    failed=any(r['state'].split()[0] in ['FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'] or (r['state']=='COMPLETED' and 'validation' not in r) for r in state['seeds'])
    state['stage_status']='complete' if complete else ('blocked' if failed else 'running')
    write(OUT/'STATUS.json',state)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':state['checked_at_utc'],'seeds':[{k:r.get(k) for k in ['seed','state','reason','epochs_saved','last_loss','local_hash_validation']} for r in state['seeds']]},ensure_ascii=False)+'\n')
    backfill(state)
    print(json.dumps({'status':state['stage_status'],'seeds':[{k:r.get(k) for k in ['seed','state','reason','epochs_saved']} for r in state['seeds']]},ensure_ascii=False),flush=True)
    return complete or failed
def backfill(state):
    label={'complete':'已完成','blocked':'受阻','running':'进行中'}[state['stage_status']]
    rows=[]
    for r in state['seeds']:
        rows.append(f"| {r['seed']} | 1539537_{r['seed']} | {r['state']} | {r['epochs_saved']}/25 | {r.get('last_loss','待记录')} | {r.get('reason','')} | {r.get('local_hash_validation','待完成')} |")
    text=f'''# A0.4 执行回填

状态：**{label}**。更新于 {state['checked_at_utc']}。唯一配置为 **X4G+D（GD）**，三个 seed 各 25 epoch，后续最终检索流程为 HNSW top1000 → task prior → top10。

## 1. 进度反馈

| seed | 作业 ID | 调度状态 | 已保存 epoch | 最近 loss | 排队原因/节点 | 本地产物哈希 |
|---:|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

训练器每 5 epoch 写 checkpoint 与原生记录，表格显示已保存进度；两次保存之间以作业和进程状态观察。每个 epoch 的 loss 均保存在最终 history 中。

## 2. 冻结配置和前置验收

A0.3 全规模 smoke 1539530 已通过。提交前核验三个正式目录均不存在；正式训练从 epoch 0 开始。完整源代码取自 smoke 已校验快照，准备记录见 [A04_PREFLIGHT.json](A0_runs/A0_4/A04_PREFLIGHT.json)。启动后逐次核验 resolved config 全部 40 键和新图 digest。

- 图：`/u801/x98liu/model_lake/data1m/a0_20260912/graph`，digest `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`。
- 独立代码：`/u801/x98liu/model_lake/a0_formal_20260913/ModelLakeFishing`。
- 正式目录：`/u801/x98liu/model_lake/runs/A0_20260912/A0GD_full_s{0,1,2}_e25`。
- 每作业资源：watgpu308，1 GPU、8 CPU、128 GiB RAM、2 小时 wall-time；Slurm 数组 1539537。
- 参数：`--rung full --seed SEED --epochs 25 --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 --skip-diagnostics --lake-gamma 0.5 --global-n-datasets 128`，graph/out/family-vocab 均显式指定绝对路径。

## 3. 产物与完成条件

每个作业完成后自动核验：25 条 history、末 checkpoint epoch 24、40 键配置、图绑定、机制门和 A0_RUN_RECORDS；固定采用 `ckpt/last.pt`。记录原生训练时间、GPU 峰值和 RSS，外层另采进程树与 GPU 使用量。结果保存到远端 `observations_s{seed}/A04_VALIDATION.json`。

本地监测程序每约 45 秒更新本页。成功作业的完整 run 自动下载到 `A0_runs/A0_4/delivery_s{seed}`，逐文件 SHA 复核后标为通过；三个 seed 全部通过才将 A0.4 标为已完成。最新原始状态和验收记录见 [STATUS.json](A0_runs/A0_4/STATUS.json)，历史更新见 [events.jsonl](A0_runs/A0_4/events.jsonl)。

## 4. 反馈区

| 事项 | 处理与反馈 |
|---|---|
| watgpu 资源排队 | 提交时 watgpu308 空余 CPU 可容纳一个作业；其余由调度器排队，上表持续显示原因 |
| 配置一致性 | 使用 smoke 已验证训练源码和同一新图；每次观察实际启动配置时核对全部 40 键 |
| 中断或失败 | 保留日志、checkpoint 和退出状态；本地监测标记受阻，续跑需按原协议记录恢复链 |
| 结果来源 | 本阶段保存新训练结果；后续检索指标在 A0.5–A0.7 计算 |
| 既有证据缺口 | 两个快照事件计数仍为 missing，后续 A0.7 继续处理 |
'''
    (DOC/'A0.4.md').write_text(text,encoding='utf-8')
    p=DOC/'A0.md';s=p.read_text(encoding='utf-8');lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 当前阶段 |'):lines[i]=f'| 当前阶段 | A0.1–A0.3 已完成；A0.4 {label}；A0.5–A0.7 待执行 |'
        elif line.startswith('| 最后更新 |'):lines[i]=f"| 最后更新 | {state['checked_at_utc']}；A0.4 {label} |"
        elif line.startswith('| 本次已完成 |'):lines[i]='| 本次已完成 | 正式三 seed 已提交，逐 seed 训练和验收状态见 [A0.4.md](A0.4.md) |'
        elif line.startswith('| 当前可做的下一步 |'):lines[i]='| 当前可做的下一步 | '+('A0.5 导出与 prior' if state['stage_status']=='complete' else '监测三个正式 run，完成后下载核验')+' |'
        elif line.startswith('| 进入训练前还缺什么 |'):lines[i]='| 进入训练前还缺什么 | 前置验收已通过，正式任务已提交；见 A0.4.md |'
        elif line.startswith('| A0.4 三 seed 从头训练 |'):lines[i]=f'| A0.4 三 seed 从头训练 | {label} | 2026-09-12 EDT 起 | [A0.4.md](A0.4.md)、[A0_4](A0_runs/A0_4/) | 数组 1539537；逐 seed 验收见回填 | 三个 25 epoch 结果验收 |'
        for r in state['seeds']:
            if line.startswith(f"| {r['seed']} | 尚未启动 |") or line.startswith(f"| {r['seed']} | 1539537_"):
                lines[i]=f"| {r['seed']} | 1539537_{r['seed']} | {r['epochs_saved']}/25，{r['state']} | 待生成 | 待计算 | 待计算 | {r.get('local_hash_validation','待验收')} |"
    s='\n'.join(lines)+'\n'
    s=s.replace('A0.1–A0.3 已完成；A0.4–A0.7 待执行，尚未启动正式训练或生成新检索结果。','A0.1–A0.3 已完成；A0.4 已提交，当前状态见反馈区；A0.5–A0.7 待执行。',1)
    p.write_text(s,encoding='utf-8')
    frozen=read(OUT/'frozen/A0_SOURCE_MANIFEST.json')
    items=[OUT/'STATUS.json',OUT/'events.jsonl',DOC/'A0.4.md',DOC/'A0.md',OUT/'train.sbatch',OUT/'monitor_train.py',OUT/'watch.py']
    if (OUT/'A04_PREFLIGHT.json').exists():items.append(OUT/'A04_PREFLIGHT.json')
    write(OUT/'MANIFEST.json',{'stage':'A0.4','status':state['stage_status'],'updated_at_utc':state['checked_at_utc'],'artifacts':[ref(p) for p in items],'prior_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json')})
    write(DOC/'A0_runs/A0_SOURCE_MANIFEST.json',{'schema':'a0.source_manifest.v1','stage':'A0.4','status':state['stage_status'],'authority':frozen['authority'],'A0_2':frozen['A0_2'],'A0_3':frozen['A0_3'],'A0_4':ref(OUT/'MANIFEST.json'),'previous_stage_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json'),'later_stages':'A0.5-A0.7 pending'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    for _ in range(960):
        try:
            done=refresh()
            if done or args.once:return
        except Exception:
            with (OUT/'watch_errors.log').open('a',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n'+traceback.format_exc())
            if args.once:raise
        time.sleep(45)
if __name__=='__main__':main()
