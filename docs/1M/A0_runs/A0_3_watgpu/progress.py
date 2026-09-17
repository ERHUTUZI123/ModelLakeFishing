from pathlib import Path
import shutil
OUT=Path(__file__).resolve().parent
DOC=OUT.parents[1]
frozen=OUT/'frozen';frozen.mkdir(exist_ok=False)
for rel in ['A0.md','A0.3.md','A0_runs/A0_SOURCE_MANIFEST.json']:
    shutil.copy2(DOC/rel,frozen/Path(rel).name)
p=DOC/'A0.3.md';s=p.read_text(encoding='utf-8')
s=s.replace('状态：**受阻。回归与信息边界测试通过；全规模 smoke 因 CUDA OOM 失败，A0.3 尚未通过验收，A0.4 未启动。**','状态：**进行中：watgpu smoke 重试作业 1539530 已提交。原本机 OOM 记录保留；A0.3 尚未通过验收，A0.4 未启动。**',1)
s+='\n## 6. watgpu 重试（2026-09-12 EDT）\n\n用户当前指令为先执行 smoke。独立代码目录 `/u801/x98liu/model_lake/a0_smoke_20260913/ModelLakeFishing`；新图 `/u801/x98liu/model_lake/data1m/a0_20260912/graph`；目标 run `/u801/x98liu/model_lake/runs/A0_20260912/smoke_s0_e1_retry1`。157 份打包文件、新图文件和词表已逐份核验；A0.2 的 152 份原源码哈希一致。\n\n- 作业 1539529：watgpu308 分配 NVIDIA L40S，47,667,740,672 字节显存；torch 2.12.0+cu130、pyg_lib 可用。因临时环境缺 pytest，47 秒后失败，未进入训练。\n- 作业 1539530：临时环境补装与本机相同的 pytest 9.1.1 后重试；仅修改调度环境准备，不改训练源码/配置。\n- 新监测覆盖 Linux 解释器进程树、OS 子进程 RSS 高水位和 nvidia-smi 进程显存采样；训练器自身另记录 PyTorch 峰值。\n- 当前等待重试作业完成；日志与脚本见 [A0_3_watgpu](A0_runs/A0_3_watgpu/)。旧的失败结论和资源缺口仅描述第一次本机尝试。\n'
p.write_text(s,encoding='utf-8')
p=DOC/'A0.md';s=p.read_text(encoding='utf-8')
s=s.replace('A0.3 测试通过、全规模 smoke 受阻','A0.3 测试通过、watgpu smoke 重试中',1)
s+='\n时间：2026-09-12 EDT\n阶段：A0.3 watgpu smoke 重试中；当前用户限定只做 smoke，A0.4 未提交。\n作业：1539529 缺 pytest 失败；1539530 补齐临时测试依赖后重试。\n输入：新图和独立源码逐份哈希一致；训练参数不变。详见 A0.3.md §6。\n'
p.write_text(s,encoding='utf-8')
print('Progress recorded; prior delivery frozen.')
