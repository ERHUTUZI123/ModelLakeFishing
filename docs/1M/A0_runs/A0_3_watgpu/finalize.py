from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
archive=OUT/'smoke_delivery.tar.gz'
assert sha(archive)=='1f9930d2d107c7cb3cd541271809727f8d31bc4699e0723db9de411bf75b98b4'
delivery=OUT/'delivery';delivery.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(delivery,filter='data')
for rel,digest in read(delivery/'RUN_FILE_HASHES.json').items():assert sha(delivery/'run'/rel)==digest,rel
r=read(delivery/'run/SMOKE_REPORT.json');v=read(delivery/'REMOTE_SMOKE_VALIDATION.json')
assert v['status']=='PASS' and r['mechanism_gate']['passed']
assert '1539530|COMPLETED|0:0|' in (delivery/'sacct_final.txt').read_text()
now=datetime.now(timezone.utc).isoformat()
p=DOC/'A0.3.md';s=p.read_text(encoding='utf-8')
s=s.replace('状态：**进行中：watgpu smoke 重试作业 1539530 已提交。原本机 OOM 记录保留；A0.3 尚未通过验收，A0.4 未启动。**','状态：**已完成。完整回归 396 项通过；watgpu 全规模 1 epoch smoke PASS，作业 1539530 为 COMPLETED / 0:0。A0.4 未启动。**\n\n最新结果见 §6；§1–§5 保留第一次本机失败及当时待办，已由此次 watgpu smoke 补齐运行和资源验收。',1)
s=s.replace('## 1. 实际结果','## 1. 第一次本机尝试（历史记录）',1)
s=s[:s.index('## 6. watgpu 重试')]+'''## 6. watgpu smoke 最终结果

执行日期：2026-09-12 EDT（作业记录为 2026-09-13 UTC）。用户本轮最终指令为仅执行 smoke，正式训练未提交。

| 项目 | 结果 |
|---|---|
| 成功作业 | **1539530，watgpu308，Slurm COMPLETED，ExitCode 0:0** |
| 设备 / 环境 | NVIDIA L40S，47,667,740,672 字节显存；Python 3.11，torch 2.12.0+cu130，pyg_lib 可用 |
| 训练规模 | 3,016,439 model；18,729 dataset；seed 0，fresh，1 epoch，batch 1024 |
| 配置 / 图 | 40 个有效配置键完全一致；图 digest 保持 `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db` |
| 优化 | **51 次 optimizer step**；31 个梯度张量全部有限且非零；参数和 optimizer state 有限 |
| 训练 loss | 23.021186342426375；单 epoch 不要求跨 epoch 下降 |
| checkpoint | epoch 0 与 last.pt 已生成，保存/加载往返校验通过 |
| smoke gate | **PASS**；未评 test，产物禁止用于初始化正式训练 |
| 远端信息边界预检 | 1 passed，4 warnings，4.25 秒 |
| 训练段耗时 | **162.9 秒**；含原训练入口 setup，不能视为纯稳定态 epoch 时间 |
| 进程墙钟 / Slurm 作业总时长 | 181.32 秒 / 237 秒（包含环境准备和预检） |
| PyTorch 峰值显存 | **37.942 GiB**（allocated） |
| nvidia-smi 进程显存峰值采样 | **44.133 GiB**，2 秒间隔；与 allocated 口径不同 |
| 训练子进程 OS RSS 高水位 | **32.611 GiB** |
| 进程树 RSS 峰值采样 | **28.292 GiB**，2 秒间隔，可能漏掉短时峰值 |
| Slurm batch MaxRSS | 39,876,344 KiB，约 **38.029 GiB**，覆盖作业环境准备等不同范围 |

训练 run：`/u801/x98liu/model_lake/runs/A0_20260912/smoke_s0_e1_retry1`。独立源码目录：`/u801/x98liu/model_lake/a0_smoke_20260913/ModelLakeFishing`。使用新图 `/u801/x98liu/model_lake/data1m/a0_20260912/graph`。

远端代码及数据处理：本地先核验 A0.2 的 152 份源文件；上传包共 157 份代码/协议/调度文件。新图的未修改文件从远端既有图复制前后分别校验 SHA，修复后的 dataset 特征、meta 和 repair report 使用本地已冻结字节；词表哈希一致。完成后再次校验除已记录调度环境补丁外的 156 份文件。未改训练源码、损失、采样、batch 或图规模。

### 6.1 失败处理与新证据

作业 **1539529** 已获得 L40S，但临时环境缺 `pytest`，47 秒后退出，未进入训练。随后只在新作业的临时环境安装与本机相同的 `pytest==9.1.1`，重投为 1539530。预检、smoke 和保存全部通过。原始失败日志保留。

此次监测覆盖 Linux 训练解释器及其子进程，记录 OS RSS 高水位和 GPU 进程采样；原训练器记录精确的 PyTorch allocated 峰值。第一次本机监测缺口已由新运行的观测补齐，不反填或修改旧测量。

证据：[SMOKE_REPORT](A0_runs/A0_3_watgpu/delivery/run/SMOKE_REPORT.json)、[训练日志](A0_runs/A0_3_watgpu/delivery/observations/train.log)、[资源报告](A0_runs/A0_3_watgpu/delivery/observations/RESOURCE_REPORT.json)、[Slurm 最终状态](A0_runs/A0_3_watgpu/delivery/sacct_final.txt)、[远端验收](A0_runs/A0_3_watgpu/delivery/REMOTE_SMOKE_VALIDATION.json)。下载包 SHA 和所有 run 文件哈希已在本地复核。

### 6.2 后续资源估算（未提交）

仅按此次训练段 162.9 秒线性外推，25 epoch 约 **67.9 分钟/seed**，三个 seed 串行约 **3.39 小时**。这是调度参考，未覆盖正式训练末尾 test/full 评估、排队和环境准备；首 epoch 自身也包含初始化开销。正式运行必须另记实际耗时，不把此估计填入结果指标。

本次申请 8 CPU、128 GiB RAM、1 张 L40S，单 epoch 已验证可运行。实测进程显存接近设备容量，后续更长运行仍须持续监测。可将正式作业初始 wall-time 设为 2 小时/seed 并监测，而不是沿用此次 30 分钟 smoke 限额；这不改变训练配方。本轮未提交这些作业。

### 6.3 最新反馈区

| 问题 | 当前结论 | 下一步 |
|---|---|---|
| 本机 CUDA OOM | 已通过 watgpu L40S 完成同配置 smoke | 正式训练使用已验证资源 |
| watgpu 缺 pytest | 临时环境补齐后预检通过 | 保留环境版本记录和作业日志 |
| 资源观测缺口 | 新运行已记录进程树、OS 和 GPU 峰值 | 正式训练继续记录 |
| A0.3 验收 | **已完成** | 当前只做 smoke，A0.4 保持待执行 |
| 两个上游事件计数缺证据 | 仍为 missing，影响 A0.7 | 后续继续处理，不使用旧值补齐 |
'''
p.write_text(s,encoding='utf-8')
p=DOC/'A0.md';s=p.read_text(encoding='utf-8')
s=s.replace('A0.1、A0.2 已完成；A0.3 测试通过、watgpu smoke 重试中','A0.1–A0.3 已完成',1)
s=s.replace('A0.3 回归通过，smoke 因 CUDA OOM 受阻','A0.3 回归与 watgpu 全规模 smoke 均通过',1)
lines=s.splitlines()
updates={
'| 最后更新 |':'| 最后更新 | 2026-09-12 EDT；A0.3 watgpu smoke PASS |',
'| 当前阶段 |':'| 当前阶段 | A0.1–A0.3 已完成；A0.4–A0.7 待执行 |',
'| 本次已完成 |':'| 本次已完成 | 完整回归 396 passed；watgpu 1539530 完整规模 1 epoch、51 步优化、checkpoint 往返校验 PASS，见 [A0.3.md](A0.3.md) §6 |',
'| 当前可做的下一步 |':'| 当前可做的下一步 | A0.4 前置已满足；本轮用户限定 smoke，未提交正式训练 |',
'| 进入训练前还缺什么 |':'| 进入训练前还缺什么 | A0.3 已通过；正式启动仍须核验三个新 run 目录并记录配置/资源 |',
'| A0.3 测试与 smoke |':'| A0.3 测试与 smoke | 已完成 | 2026-09-12 EDT | [A0.3.md](A0.3.md)、[watgpu 证据](A0_runs/A0_3_watgpu/) | 396 passed；smoke 1539530 COMPLETED 0:0、gate PASS | A0.4 待执行 |'}
for i,line in enumerate(lines):
    for prefix,new in updates.items():
        if line.startswith(prefix):lines[i]=new
    if line.startswith('| A0.2 新图与实现适配 |'):lines[i]=line.replace('A0.3 smoke 受阻','A0.3 已完成')
s='\n'.join(lines)+'\n\n时间：'+now+'\n阶段：A0.3 已完成。watgpu 作业 1539530，1 epoch、51 步优化、checkpoint 往返校验 PASS；Slurm COMPLETED 0:0。训练段 162.9 秒，PyTorch 峰值 37.942 GiB。当前用户仅要求 smoke，A0.4 未启动。证据见 A0.3.md §6。\n'
old=(OUT/'frozen/A0.md').read_text(encoding='utf-8')
assert old[old.index('## 1.'):old.index('## 10.')]==s[s.index('## 1.'):s.index('## 10.')]
p.write_text(s,encoding='utf-8')
with (DOC/'A0_runs/A0_COMMANDS.md').open('a',encoding='utf-8') as f:f.write('\n\n## A0.3 watgpu smoke（完成）\n\n实际 SSH/bash 命令脚本及其日志见 `A0_3_watgpu/`。首作业 1539529 缺 pytest；在临时环境安装 pytest==9.1.1 后作业 1539530 成功。逐字训练 argv 见 `A0_3_watgpu/delivery/observations/command.json`；实际 sbatch 见 `delivery/executed_smoke.sbatch`。全部 40 键保持一致，未提交 A0.4。\n')
with (DOC/'A0_runs/A0_DISCREPANCIES.md').open('a',encoding='utf-8') as f:f.write('\n\n## A0.3 watgpu 解决资源阻断\n\n1539530 在 L40S 上完成同配置 1 epoch，gate PASS、checkpoint 往返通过，Slurm COMPLETED 0:0。新监测补齐资源观察，旧 OOM/无效 RSS 记录保留。远端 Python 3.11、torch 2.12.0+cu130 与本机 Python 3.13、cu126 环境不同，远端信息边界测试通过，环境已归档。唯一调度补丁为临时安装 pytest==9.1.1；训练源码和 40 键配置不变。A0.4 未启动。\n')
write(OUT/'LOCAL_DELIVERY_VALIDATION.json',{'status':'PASS','checked_at_utc':now,'run_files_verified':len(read(delivery/'RUN_FILE_HASHES.json')),'archive':ref(archive),'config_keys_equal':40,'job_id':1539530,'slurm_status':'COMPLETED 0:0','plan_sections_1_to_9_unchanged':True,'formal_training_started':False})
files=[p for p in OUT.rglob('*') if p.is_file()]+[DOC/'A0.3.md',DOC/'A0.md',DOC/'A0_runs/A0_COMMANDS.md',DOC/'A0_runs/A0_DISCREPANCIES.md']
write(OUT/'MANIFEST.json',{'stage':'A0.3','status':'complete','created_at_utc':now,'artifacts':[ref(p) for p in files],'previous_attempt':ref(DOC/'A0_runs/A0_3/A0_3_MANIFEST.json'),'self_hash_policy':'Excludes itself.'})
prior=read(OUT/'frozen/A0_SOURCE_MANIFEST.json')
write(DOC/'A0_runs/A0_SOURCE_MANIFEST.json',{'schema':'a0.source_manifest.v1','stage':'A0.3','status':'complete','created_at_utc':now,'authority':prior['authority'],'previous_stage_manifest':ref(OUT/'frozen/A0_SOURCE_MANIFEST.json'),'A0_2':prior['A0_2'],'A0_3':ref(OUT/'MANIFEST.json'),'later_stages':'A0.4-A0.7 pending; user scoped current turn to smoke only. No formal training or retrieval results.'})
for item in read(OUT/'MANIFEST.json')['artifacts']:assert ref(Path(item['path']))==item
print(json.dumps({'status':'PASS','stage':'A0.3 complete','artifacts_verified':len(files),'training_seconds':r['wallclock_s'],'formal_training_started':False}))
