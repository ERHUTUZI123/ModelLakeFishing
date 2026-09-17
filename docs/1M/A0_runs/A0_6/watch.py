"""Monitor A0.6, retrieve raw measurements, and backfill evidence documents."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess,time,tarfile,traceback
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1];HOST='x98liu@watgpu.cs.uwaterloo.ca'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(p)
SCRIPT='''python3 - <<'PY'
from pathlib import Path
import subprocess,json,hashlib,tarfile
base=Path('/u801/x98liu/model_lake/a0_eval_20260914');out=base.parent/'data1m/a0_20260912/metrics'
job=(base/'job_id.txt').read_text().strip()
def cmd(args):return subprocess.run(args,capture_output=True,text=True).stdout
acct=cmd(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,MaxRSS','-n','-P'])
queue=cmd(['squeue','-j',job,'-h','-o','%i|%T|%R'])
s={'job_id':job,'state':'UNKNOWN','sacct':acct,'queue':queue,'start_estimate':cmd(['squeue','--start','-j',job])}
for line in acct.splitlines():
    a=line.split('|')
    if a[0]==job:s.update(state=a[1],exit_code=a[2],elapsed=a[3])
for line in queue.splitlines():
    a=line.split('|')
    if a[0]==job:s.update(state=a[1],reason=a[2])
for name in ['exact','hnsw','finalize',f'slurm-{job}']:
    p=base/(name+'.log')
    if p.exists():s[name+'_tail']=p.read_text(errors='replace').splitlines()[-7:]
for p,key in [(out/'A0_EVALUATION_REPORT.json','report'),(base/'A06_VALIDATION.json','validation')]:
    if p.exists():
        try:s[key]=json.loads(p.read_text())
        except json.JSONDecodeError:pass
if s['state']=='COMPLETED' and s.get('validation',{}).get('status')=='PASS':
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    bundle=base/'audit_delivery.tar.gz'
    if not bundle.exists():
        manifest=json.loads((out/'A0_EVALUATION_MANIFEST.json').read_text())
        for name,r in manifest['artifacts'].items():assert sha(out/name)==r['sha256'],name
        records={}
        with tarfile.open(bundle,'w:gz') as tar:
            for p in out.iterdir():
                if p.is_file() and p.suffix!='.bin':
                    name='metrics/'+p.name;tar.add(p,arcname=name);records[name]={'sha256':sha(p),'bytes':p.stat().st_size}
            for p in base.iterdir():
                if p.is_file() and p.suffix in ['.json','.time','.log','.txt','.sbatch','.py']:
                    name='operations/'+p.name;tar.add(p,arcname=name);records[name]={'sha256':sha(p),'bytes':p.stat().st_size}
            p=base/'DELIVERY_HASHES.json';p.write_text(json.dumps(records,indent=2)+chr(10));tar.add(p,arcname=p.name)
    s['audit_bundle']={'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size}
print(json.dumps(s))
PY
'''
def refresh():
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'bash','-s'],input=SCRIPT.encode(),capture_output=True,timeout=240)
    if p.returncode:raise RuntimeError(p.stdout.decode(errors='replace')+p.stderr.decode(errors='replace'))
    s=json.loads(p.stdout);s['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    if 'audit_bundle' in s:
        target=OUT/'audit_delivery.tar.gz'
        if not target.exists():
            p=subprocess.run(['scp','-q','-o','BatchMode=yes',HOST+':'+s['audit_bundle']['path'],str(target)],capture_output=True,timeout=240)
            assert p.returncode==0,p.stderr
        assert sha(target)==s['audit_bundle']['sha256']
        delivery=OUT/'delivery'
        if not delivery.exists():
            delivery.mkdir()
            with tarfile.open(target) as tar:tar.extractall(delivery,filter='data')
        for name,r in read(delivery/'DELIVERY_HASHES.json').items():assert sha(delivery/name)==r['sha256'] and (delivery/name).stat().st_size==r['bytes']
        s['local_hash_validation']='PASS'
    fail=s['state'].split()[0] in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'] or (s['state']=='COMPLETED' and 'audit_bundle' not in s)
    s['stage_status']='complete' if s.get('local_hash_validation')=='PASS' else ('blocked' if fail else 'running')
    local=OUT/'local_verification/STATUS.json'
    if local.exists():
        s['local_compute_verification']=read(local)
        if s['local_compute_verification']['status']=='FAIL':s['stage_status']='blocked'
        elif s['stage_status']=='complete' and s['local_compute_verification']['status']!='PASS':s['stage_status']='running'
    write(OUT/'STATUS.json',s)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':s['updated_at_utc'],'state':s['state'],'stage':s.get('report',{}).get('stage'),'exact_seeds':list(s.get('report',{}).get('per_seed',{})),'hnsw_seeds':list(s.get('report',{}).get('hnsw',{}))})+'\n')
    backfill(s)
    print(json.dumps({'status':s['stage_status'],'job':s['job_id'],'state':s['state'],'reason':s.get('reason'),'stage':s.get('report',{}).get('stage'),'exact_seeds':list(s.get('report',{}).get('per_seed',{})),'hnsw_seeds':list(s.get('report',{}).get('hnsw',{}))}),flush=True)
    return s['stage_status'] in ['complete','blocked']
def backfill(s):
    label={'complete':'已完成','blocked':'受阻','running':'进行中'}[s['stage_status']]
    report=s.get('report',{});rows=[]
    for seed in range(3):
        exact=report.get('per_seed',{}).get(str(seed),{});h=report.get('hnsw',{}).get(str(seed),{})
        metric=h.get('rows',{}).get('G_hnsw1000_task',{})
        rows.append(f"| {seed} | {'完成' if exact else '待完成'} | {'完成' if h else '待完成'} | {metric.get('gold@10','待测量')} | {h.get('recall@1000','待校准')} | {h.get('ef_search','待校准')} | {h.get('build_seconds','待测量')} |")
    summary=json.dumps({k:report[k] for k in ['hnsw_summary','latency_summary_ms','index_cost','correctness_gates','effectiveness_gates','ann_fidelity_status'] if k in report},ensure_ascii=False,indent=2)
    local=s.get('local_compute_verification',{})
    local_note=f"RTX 4060 Laptop 分块校验向量和全部已保存候选分数；i7 使用 8 个计算线程、2 个哈希工作线程核验文件、融合公式与 top10。当前 {local.get('status','待启动')} / {local.get('phase','待启动')}；详见 [本机反馈](A0_runs/A0_6/local_verification/STATUS.json)。本机验证独立记录，论文评测计时取自远端原流程。"
    text=f'''# A0.6 执行回填

状态：**{label}**。更新于 {s['updated_at_utc']}。作业 **{s['job_id']}**：`{s['state']}`，原因/节点：`{s.get('reason','见最终状态')}`；当前 evaluator 阶段：`{report.get('stage','环境准备/排队')}`。

唯一最终方法：**X4G+D → HNSW top1000 → task prior → top10**。

## 1. 逐 seed 进度与实测结果

| seed | exact | HNSW/计时 | 最终 gold@10 | recall@1000 | ef_search | 建索引秒 |
|---:|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 2. 执行协议

严格按 A0.md 顺序：三个 seed 的新 exact 全部完成 → 各 seed 新建 HNSW 索引、ef 校准及计时 → finalize。复用 A0.4 已冻结源码和 A0.5 新导出向量及 split-specific sidecar。输入产物实际 SHA 在提交前全部核验，评测入口进一步核验图、25 epoch checkpoint、gold/query 身份、行序和 prior 边界。

固定配置：K=1000、返回 10；M=32、ef_construction=200；建索引线程 8；ef 网格 1000/1500/2000/3000/5000，按首次 recall≥0.99 停止；exact query/model chunk=16/50000。测量以预计算 query embedding、预热、单查询 HNSW 线程执行，核验和额外写盘位于计时区间外。

输入：`/u801/x98liu/model_lake/data1m/a0_20260912/exports/A0GD_full_s{0,1,2}_e25`，对应 sidecar 同目录。输出：`/u801/x98liu/model_lake/data1m/a0_20260912/metrics`。初始输出目录不存在。实际命令见 [run.sbatch](A0_runs/A0_6/run.sbatch) 和 [retry_308.sh](A0_runs/A0_6/retry_308.sh)。

## 3. Linux 路径迁移记录

原协议及审计记录是 Windows 文件路径。为保持评测源码不变，本次显式传入 `--a0-protocol-file .../inputs/A0_PROTOCOL.linux.json` 和 `--a0-query-identity .../inputs/audit/A0_QUERY_IDENTITY.jsonl`。

迁移副本修改了协议的 6 个 path、输入审计的 95 个 path，并更新协议对审计副本的 SHA/bytes 引用。深层比较确认其余配置、规则、测量值、gold 身份和所有数据文件哈希相同；原文件保存在本轮 frozen 目录。说明及完整差异见 [PATH_RELOCATION.json](A0_runs/A0_6/PATH_RELOCATION.json)。

## 4. 验收与产物

各新 index、exact/HNSW 原始候选、prior/fused/top10、独立 gold、校准尝试、query 计时数组和资源记录写入 `A0_EVALUATION_MANIFEST.json`。finalize 校验绑定产物，单独记录正确性与效果门槛。原效果门槛未达到时仍如实保留测量；A0.7 负责全部指标的独立复算。

本阶段交付校验还检查：每 query 的 1000 个候选唯一、query ID 在 exact/HNSW 间一致、score/prior/fused 有限、计时分量之和等于 total、建索引时间大于 0、三个 seed 均完成。三个阶段成功后自动下载完整原始测量与报告，并按逐文件 SHA 复核。大体积 `.bin` 索引留在上述 watgpu metrics 目录，哈希记录随报告回收。

本地交付目录：`A0_runs/A0_6/delivery`；当前本地哈希验收：'''+s.get('local_hash_validation','待完成')+'''。

## 5. 当前汇总（来自本轮 evaluator）

```json
'''+summary+'''
```

## 6. 反馈区

| 事项 | 实际反馈 |
|---|---|
| 608 排队 | 原作业 1540916 预计等待数小时，仍 PENDING 时取消 |
| 308 环境预检 | 作业 1540917 因缺 hnswlib 退出，39 秒，exact 尚未进入；原日志保留 |
| 当前资源 | 在 watgpu308 重投 1540930；临时环境补装与本地相同的 hnswlib==0.8.0，版本写入 pip_freeze；重投见 retry_hnswlib.sh |
| 最新资源检查 | 作业 1540930 已在 watgpu308 的 L40S 获得资源；实时状态见本页开头与 STATUS.json |
| 执行资源 | 1 GPU、8 CPU、128 GiB RAM、2 小时时限；计算参数一致 |
| 路径兼容 | 生成有完整差异记录的 Linux 协议/审计副本；原冻结文件留存 |
| 后台监测 | 每约 45 秒回填；失败时保留日志与阶段状态，成功后下载核验 |
| 既有证据缺口 | 两个上游事件计数仍为 missing，A0.7 全指标验收继续处理 |

证据：[STATUS.json](A0_runs/A0_6/STATUS.json)、[events.jsonl](A0_runs/A0_6/events.jsonl)、[交付校验脚本](A0_runs/A0_6/validate.py)。
'''
    if local:text+='\n## 7. 本机 4060 / i7 并行验收反馈\n\n'+local_note+'\n\n跨设备分数校验阈值为绝对误差 1e-5，仅用于浮点一致性诊断；保存的分数、排序和效果门槛保持原值。三个 seed 的 exact/HNSW 各 4,122,000 个候选分数将逐项验证。A0.7 的全部指标独立复算仍单列待执行。\n'
    (DOC/'A0.6.md').write_text(text,encoding='utf-8')
    p=DOC/'A0.md';lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 当前阶段 |'):lines[i]=f'| 当前阶段 | A0.1–A0.5 已完成；A0.6 {label}；A0.7 待执行 |'
        elif line.startswith('| 最后更新 |'):lines[i]=f"| 最后更新 | {s['updated_at_utc']}；A0.6 {label} |"
        elif line.startswith('| 本次已完成 |'):lines[i]='| 本次已完成 | A0.6 输入与路径迁移核验、作业提交；逐 seed 测量见 [A0.6.md](A0.6.md) |'
        elif line.startswith('| 当前是否已有新结果 |'):lines[i]='| 当前是否已有新结果 | 本轮三个 seed 的训练与导出已完成；exact/HNSW 新测量进度及最终结果见 [A0.6.md](A0.6.md) |'
        elif line.startswith('| 当前可做的下一步 |'):lines[i]='| 当前可做的下一步 | '+('A0.7 独立复算全部指标' if s['stage_status']=='complete' else '监测 exact → HNSW → finalize，完成后核验交付')+' |'
        elif line.startswith('| A0.6 全部评测与计时 |'):lines[i]=f"| A0.6 全部评测与计时 | {label} | 2026-09-13 EDT 起 | [A0.6.md](A0.6.md)、[A0_6](A0_runs/A0_6/) | 作业 {s['job_id']}，测量状态见回填 | A0.7 独立复算 |"
    current='\n'.join(lines)+'\n';old=(OUT/'frozen/A0.md').read_text(encoding='utf-8')
    assert current[current.index('## 1.'):current.index('## 10.')]==old[old.index('## 1.'):old.index('## 10.')]
    p.write_text(current,encoding='utf-8')
    previous=read(OUT/'frozen/A0_SOURCE_MANIFEST.json')
    items=[DOC/'A0.6.md',DOC/'A0.md',OUT/'STATUS.json',OUT/'events.jsonl',OUT/'run.sbatch',OUT/'validate.py',OUT/'watch.py',OUT/'PATH_RELOCATION.json',OUT/'inputs.tar.gz',OUT/'retry_308.sh']
    if (OUT/'audit_delivery.tar.gz').exists():items.append(OUT/'audit_delivery.tar.gz')
    for p in [OUT/'local_verify.py',OUT/'local_verification/STATUS.json',OUT/'local_verification/EXACT_MANIFEST.remote.json',OUT/'frozen/watch_before_local.py']:
        if p.exists():items.append(p)
    write(OUT/'MANIFEST.json',{'stage':'A0.6','status':s['stage_status'],'updated_at_utc':s['updated_at_utc'],'artifacts':[ref(p) for p in items],'previous_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json')})
    new={k:previous[k] for k in ['authority','A0_2','A0_3','A0_4','A0_5']}
    new.update(schema='a0.source_manifest.v1',stage='A0.6',status=s['stage_status'],A0_6=ref(OUT/'MANIFEST.json'),previous_stage_manifest=ref(OUT/'frozen/A0_SOURCE_MANIFEST.json'),later_stages='A0.7 pending')
    write(DOC/'A0_runs/A0_SOURCE_MANIFEST.json',new)
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    for _ in range(2880):
        try:
            if refresh() or args.once:return
        except Exception:
            with (OUT/'watch_errors.log').open('a',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n'+traceback.format_exc())
            if args.once:raise
        time.sleep(45)
if __name__=='__main__':main()
