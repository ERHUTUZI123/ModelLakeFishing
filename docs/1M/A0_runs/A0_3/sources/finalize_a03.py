"""Backfill A0.3 evidence without modifying the training recipe."""
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
DOC=ROOT/'docs/1M'
RUN=Path('D:/research/model_lake/runs/A0_20260912/smoke_s0_e1')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def ref(p):return {'path':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

assert not (OUT/'A0_3_MANIFEST.json').exists()
cfg=read(RUN/'metadata/resolved_config.json')
assert cfg['resolved_config']==read(OUT/'frozen/A0_PROTOCOL.json')['required_config_audit']['resolved_config']
assert len(cfg['resolved_config'])==40
assert cfg['graph_sha256']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
suite=ET.parse(OUT/'regression.xml').getroot()
cases=list(suite.iter('testcase'))
assert len(cases)==396 and not any(list(suite.iter(k)) for k in ['failure','error','skipped'])
smoke=read(OUT/'smoke.json');assert smoke['returncode']==1
assert 'torch.OutOfMemoryError' in (OUT/'smoke.log').read_text(encoding='utf-8')
assert not list(RUN.rglob('*.pt')) and not list(RUN.rglob('SMOKE_REPORT*'))
now=datetime.now(timezone.utc).isoformat()
report='''# A0.3 执行回填

状态：**受阻。回归与信息边界测试通过；全规模 smoke 因 CUDA OOM 失败，A0.3 尚未通过验收，A0.4 未启动。**

执行日期：2026-09-12（EDT）。遵循 [A0.md](A0.md)，最终系统保持 **X4G+D → HNSW top1000 → task prior → top10**。

## 1. 实际结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 前置核验 | A0.2 manifest 和 152 份源文件哈希一致 | [入口记录](A0_runs/A0_3/A0_3_ENTRY.json) |
| 新增信息边界测试 | 1 passed，已包含在下方 396 项中 | [日志](A0_runs/A0_3/boundary_3.log) |
| 原定完整回归 | **396 passed，349 warnings，0 failures，0 skipped**；pytest 23.47 秒，命令墙钟约 26.07 秒 | [日志](A0_runs/A0_3/regression.log)、[JUnit](A0_runs/A0_3/regression.xml) |
| 全规模 smoke | fresh、seed 0、目标 1 epoch；62.18 秒后退出码 1，首个优化步骤之前 OOM | [实际命令与记录](A0_runs/A0_3/smoke.json)、[日志](A0_runs/A0_3/smoke.log) |
| 配置冻结 | 实际 resolved config 全部 40 个键与 A0_PROTOCOL 一致 | [核验记录](A0_runs/A0_3/A0_3_VALIDATION.json) |
| 全规模 checkpoint / 完成报告 | 均未生成；完整规模反向传播、更新和保存能力仍待验证 | 同上 |
| 正式训练与新检索指标 | 未启动 / 未生成 | 三个正式 seed 保持待执行 |

完整回归：`.venv\\Scripts\\python.exe -X utf8 -B -m pytest scale1m/tests stage2TrainGraphSAGE/tests -q`。实际命令另带 JUnit 输出参数，见 [regression.json](A0_runs/A0_3/regression.json)。

## 2. 信息边界与一致性

新增 [test_a03_information_boundary.py](../../scale1m/tests/test_a03_information_boundary.py) 在 120 个 model、24 个 dataset 的小型 fixture 上调用实际 GD 训练入口，完成同配置的一轮优化对照。固定节点、边端点与顺序、split 和评估查询，仅扰动 test-root 性能权重及供特征生成使用的 eligibility。

- 修复后的输入一致；train/val/test root 分离，held-out 性能边及反向边排除。
- 全图性能 lookup 发生变化，但训练实际取用的监督、global proposal 的 q/logq 不变。
- 固定 RNG 后，两次训练历史、model/scorer 参数及 held-out forward 完全一致。
- 输出 128 维、有限值、L2 norm 检查通过；分块与完整推理最大绝对误差小于 `1e-5`。

其余要求由完整套件覆盖：`test_checkpoint.py` 验证恢复，`test_prepare_a0_graph.py` 和 `test_a0_producer_bindings.py` 验证图/模型绑定，`test_t0_scale.py` 和 `test_export_rf.py` 验证分块与行序，评测及复算相关测试验证 prior、候选、融合和独立重算。这些为测试证据，全规模产物验证仍待 smoke 成功。

新增测试初次因包路径失败，第二次因多线程 CPU 浮点归约导致逐位比较失败。修正测试导入及 fixture 单线程环境后通过；保留 `boundary.log`、`boundary_2.log`，未修改生产算法或放宽数值阈值。

## 3. 失败与资源记录

运行目录：`D:\\research\\model_lake\\runs\\A0_20260912\\smoke_s0_e1`。输入图 digest：`acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`。

设备 RTX 4060 Laptop，8 GiB 显存；系统 RAM 约 15.78 GiB，入口可用约 4.41 GiB，D 盘空闲约 211.94 GiB。模型节点 3,016,439；单份 float32 特征 `[3016439,448]` 约 5.04 GiB。

失败位置：[train.py](../../stage2TrainGraphSAGE/train.py) 第 129 行 `g_base = train_data.clone().to(device)`。错误报告申请 5.04 GiB 失败，同时报告 GPU 空闲 6.86 GiB，因此不能简单解释为当时空闲显存小于申请量，具体分配失败机制尚未确定。

代码确认 global loss 分支做全图前向，每步还会 `g_base.clone()`。仅两份模型特征就约 10.07 GiB，另有图边、类别特征、激活和梯度；原实现此路径无法仅凭 fanout 或推理分块适配 8 GiB 显存。本次保持 batch、N、损失、采样与精度不变。

**监测缺口：** 脚本只监测直接子进程，Windows venv 启动器可能另启解释器，约 4 MiB working set 不能代表训练峰值，标为无效。系统可用 RAM 最低采样约 304.41 MiB，提示系统内存压力。训练 RSS 峰值、GPU 峰值没有可靠记录。62.18 秒是失败耗时，**不能作为 epoch 耗时或估算 25 epoch / 三 seed 总耗时**。

## 4. 下一步（尚未执行）

1. 在具备更大显存和主存的资源上，保持同一图和全部 40 键配置；所需峰值仍须实测，当前不承诺某规格足够。
2. 重试前修正监测，覆盖完整解释器进程树和 GPU 峰值。现有 `run_a03.py` 的峰值数据不能用于资源验收。
3. 保留失败目录；在新目录 `smoke_s0_e1_retry1` 按原参数重跑。完成 1 epoch、有限梯度/参数、真实优化更新、checkpoint 保存及资源记录后，才将 A0.3 标为已完成。

```powershell
# 在合适训练资源上设置 A0.md 的路径变量；重试尚未执行。
$A0SmokeRetry = Join-Path $A0RunsRoot 'smoke_s0_e1_retry1'
if (Test-Path -LiteralPath $A0SmokeRetry) { throw 'Retry directory already exists' }
& $A0Python -m scale1m.train_rung `
  --rung full --graph $A0Graph --out $A0SmokeRetry --seed 0 --epochs 1 `
  --family-vocab $A0FamilyVocab --fanout --sparse-M --contrast-n-neg 256 `
  --chunked-infer 50000 --skip-diagnostics --lake-gamma 0.5 `
  --global-n-datasets 128 --smoke-only
if ($LASTEXITCODE -ne 0) { throw 'A0.3 smoke failed' }
```

## 5. 反馈区

| 问题 | 影响 | 处理 / 下一步 | 状态 |
|---|---|---|---|
| 全图 GPU 分配失败 | 阻止 A0.3 验收和 A0.4 | 保留日志，合适资源上按原配置重试 | 受阻 |
| 监测未覆盖训练解释器 | RSS/GPU 峰值缺失 | 重试前修正监测 | 待处理 |
| 两个上游快照事件计数缺原始证据 | 影响 A0.7 完整验收 | 继续保持 A0.2 的 missing，不复制旧值 | 未解决 |
| 用户要求减少 usage | 执行效率 | 单 agent；完整回归通过后未重复跑，未盲目重试相同资源 | 已执行 |

本轮新增测试和执行记录；生产实现与协议保持 A0.2 版本。归档哈希见 [A0_3_MANIFEST.json](A0_runs/A0_3/A0_3_MANIFEST.json)。
'''
(DOC/'A0.3.md').write_text(report,encoding='utf-8')
plan=(DOC/'A0.md').read_text(encoding='utf-8')
before=plan.split('## 10.')[0]
plan=plan.replace('A0.1、A0.2 已完成；A0.3–A0.7 待执行，尚未提交训练或生成新检索结果。','A0.1、A0.2 已完成；A0.3 测试通过、全规模 smoke 受阻；A0.4–A0.7 待执行，尚未启动正式训练或生成新检索结果。',1)
plan=plan.replace('新图和必要入口已核验；下一步为 **A0.3**。','新图和必要入口已核验；A0.3 回归通过，smoke 因 CUDA OOM 受阻，详见 [A0.3.md](A0.3.md)。',1)
lines=plan.splitlines()
replace={
'| 最后更新 |':'| 最后更新 | 2026-09-12 EDT；A0.3 回填，smoke 受阻 |',
'| 当前阶段 |':'| 当前阶段 | A0.1、A0.2 已完成；A0.3 受阻；A0.4–A0.7 待执行 |',
'| 本次已完成 |':'| 本次已完成 | A0.3 信息边界测试及完整回归 396 passed；全规模 smoke 已尝试并保留失败记录，见 [A0.3.md](A0.3.md) |',
'| 当前可做的下一步 |':'| 当前可做的下一步 | 合适资源上按原配置重试 A0.3 smoke，重试前修正进程树/GPU 监测 |',
'| 进入训练前还缺什么 |':'| 进入训练前还缺什么 | 全规模单 epoch smoke、checkpoint 和完整资源记录通过；三个正式 run 未启动 |',
'| A0.3 测试与 smoke |':'| A0.3 测试与 smoke | 受阻 | 2026-09-12 21:48 EDT / 21:53 EDT（smoke 失败） | [A0.3.md](A0.3.md)、[A0_3](A0_runs/A0_3/) | 396 passed；40 键一致；首步前 CUDA OOM，无 checkpoint | 合适资源上原配置重试 |'}
for i,line in enumerate(lines):
    for prefix,new in replace.items():
        if line.startswith(prefix):lines[i]=new
    if line.startswith('| A0.2 新图与实现适配 |'):lines[i]=line.replace('A0.3 待执行','A0.3 smoke 受阻')
plan='\n'.join(lines)+'\n'
plan+='\n时间：'+now+'\n阶段：A0.3 受阻。\n完成：信息边界及完整回归 396 passed；真实全规模 smoke 62.18 秒后 CUDA OOM，首次优化前退出，无 checkpoint。\n配置：全部 40 键与冻结协议一致；生产实现未修改，A0.4 未启动。\n缺口：本机资源不足；训练 RSS/GPU 峰值未可靠采得。下一步在合适资源上修正监测后原配置重试。\n证据：[A0.3.md](A0.3.md)、[A0_3](A0_runs/A0_3/)。\n'
# Only status text preceding section 1 and feedback section may change.
old=(OUT/'frozen/A0.md').read_text(encoding='utf-8')
assert old[old.index('## 1.'):old.index('## 10.')]==plan[plan.index('## 1.'):plan.index('## 10.')]
(DOC/'A0.md').write_text(plan,encoding='utf-8')
with (DOC/'A0_runs/A0_COMMANDS.md').open('a',encoding='utf-8') as f:f.write('\n\n## A0.3 实际执行（2026-09-12 EDT）\n\n新增边界测试、完整回归、全规模 smoke 的逐字 argv、退出码、耗时见 `A0_3/boundary_3.json`、`A0_3/regression.json`、`A0_3/smoke.json`。回归 396 passed；smoke 首步前 CUDA OOM，未启动正式训练。后续新资源重试命令在 A0.3.md §4，尚未执行。\n')
with (DOC/'A0_runs/A0_DISCREPANCIES.md').open('a',encoding='utf-8') as f:f.write('\n\n## A0.3 资源阻断与记录缺口\n\n40 键配置一致。全规模 smoke 在 train.py:129 CUDA OOM，0 个优化步骤、无 checkpoint。8 GiB GPU 无法容纳现有 global 分支双份模型特征（约 10.07 GiB，尚不含其他张量）。未改变算法或配置，A0.4 保持待执行。采样器只覆盖启动进程，训练 RSS/GPU 峰值标为 missing；完整 epoch 耗时及总训练估算也为 missing。失败日志保留，详见 A0.3.md。\n')
write(OUT/'A0_3_VALIDATION.json',{'stage':'A0.3','status':'blocked','checked_at_utc':now,'regression_passed':396,'regression_failed':0,'regression_skipped':0,'config_keys_equal':40,'graph_sha256':cfg['graph_sha256'],'smoke_returncode':1,'smoke_optimizer_steps_completed':0,'smoke_checkpoint_count':0,'smoke_report_present':False,'peak_training_rss_bytes':None,'peak_gpu_bytes':None,'epoch_seconds':None,'estimated_25_epoch_seconds':None,'monitoring_limitation':'Direct launcher process only; RSS cannot represent training interpreter. No reliable GPU peak.','min_system_available_ram_bytes':min(s['available_ram'] for s in smoke['samples']),'formal_training_started':False,'plan_sections_1_through_9_unchanged':True})
archive=OUT/'sources';archive.mkdir()
for p in [ROOT/'scale1m/tests/test_a03_information_boundary.py',OUT/'run_a03.py',Path(__file__)]:shutil.copy2(p,archive/p.name)
for p in [RUN/'metadata/resolved_config.json',RUN/'SMOKE_ONLY.json']:shutil.copy2(p,OUT/p.name)
files=[p for p in OUT.rglob('*') if p.is_file()]+[DOC/'A0.3.md',DOC/'A0.md',DOC/'A0_runs/A0_COMMANDS.md',DOC/'A0_runs/A0_DISCREPANCIES.md']
write(OUT/'A0_3_MANIFEST.json',{'stage':'A0.3','status':'blocked','created_at_utc':now,'previous_stage':ref(DOC/'A0_runs/A0_2/A0_2_MANIFEST.json'),'artifacts':[ref(p) for p in files],'smoke_run_files':[ref(p) for p in RUN.rglob('*') if p.is_file()],'self_hash_policy':'Excludes itself.'})
prior=read(OUT/'frozen/A0_SOURCE_MANIFEST.json')
write(DOC/'A0_runs/A0_SOURCE_MANIFEST.json',{'schema':'a0.source_manifest.v1','stage':'A0.3','status':'blocked','created_at_utc':now,'authority':prior['authority'],'previous_stage_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json'),'A0_2':prior['A0_2'],'A0_3':ref(OUT/'A0_3_MANIFEST.json'),'later_stages':'A0.3 smoke blocked by CUDA OOM; A0.4-A0.7 pending. No formal training or retrieval results.'})
for item in read(OUT/'A0_3_MANIFEST.json')['artifacts']:assert ref(Path(item['path']))==item
print(json.dumps({'status':'blocked','regression':396,'config_keys_equal':40,'artifacts_verified':len(files),'document':str(DOC/'A0.3.md')},ensure_ascii=False))
