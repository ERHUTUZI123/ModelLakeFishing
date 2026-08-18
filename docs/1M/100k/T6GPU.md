# T6 在 watGPU 上怎么跑

上级 runbook：[`100kplan.md`](100kplan.md) §9。上游：[`T5.md`](T5.md)（`hgraph_100k.pt`）。

你要做的事：先跑一条探测命令（§2），然后按 §5 的四步依次提交作业。
第一步（12K 锚点）没过就停下——后面三步全建立在它之上。

---

## 1. 已经写好并验过什么

代码是本地写的、本地验的，训练本身要 GPU，但验证 checkpoint/resume 不需要。

| 文件 | 干什么 |
|---|---|
| [`scale1m/train_rung.py`](../../scale1m/train_rung.py) | 训练驱动：run 目录、元数据、checkpoint/resume、机制门、MANIFEST |
| [`scale1m/checkpoint.py`](../../scale1m/checkpoint.py) | §9.3 的 checkpoint 契约：原子写、保留策略、resume 解析顺序、绑定校验 |
| [`scale1m/tests/test_checkpoint.py`](../../scale1m/tests/test_checkpoint.py) | 18 个测试（仓库合计 130 passed） |
| [`scripts/watgpu/train_rung.sbatch`](../../scripts/watgpu/train_rung.sbatch) | 作业脚本，`RUNG`/`SEED`/`EPOCHS` 三个环境变量 |

改动了两个既有文件，都是加可选参数、默认值保持原行为：
`stage2TrainGraphSAGE/train.py` 加了 `resume_state` / `history0` / `on_epoch_end`，
`ablation.py` 把这三个从 cfg 透传进去。仓库原有的 66 个 stage2 测试全部照常通过。

**配置一个字没改。** §9.1 要求原样沿用 `l1l3b_config()`，这份代码只加了集群跑作业需要的东西。
这不是洁癖：R0 锚点要复现 `gold@10 = 0.4159`，如果这里顺手动了一个超参，
锚点一偏就分不清是超参还是 T0 的那些开关造成的。

### 1.1 本地验过的三件事

**resume 和不中断的运行数值一致。** 12K 图上跑两次：一次直接 4 epoch，一次跑 2 epoch
杀掉再 resume 到 4 epoch，逐 epoch loss 对比：

```
resumed : [18.328344, 16.086379, 15.294966, 15.174848]
straight: [18.328344, 16.086379, 15.294966, 15.174849]
最大差 6.9e-07（float32 在 batch 上求均值的舍入）
```

第一版不是这样，差 1.3%。原因值得记下来：`train_eval_one` 建模型时会
`torch.manual_seed(init_seed)`，把驱动侧恢复好的 RNG 又覆盖掉了，于是 resume 之后的
batch 顺序、edge dropout、负样本全都换了一条随机路径。**RNG 必须在 `train()` 里面、
在那次 seed 之后恢复**，现在就是这么做的。这条错误不会抛异常，只会让续跑的曲线和原曲线
悄悄分叉——正是 G-W4 要拦的那类。

**机制门（G-C1）能跑出来**：12K / 2 epoch，loss 18.33 → 16.09，无 NaN，两张 embedding 表非零。

**显存和时间的一个参照点**：12K、4 epoch、RTX 4060、`--fanout --sparse-M --contrast-n-neg 256`
⇒ 峰值显存 0.643 GB，墙钟 115 秒（约 29 秒/epoch）。这个数只能当量级参考，见 §6。

---

## 2. 先跑这条探测：`pyg-lib` 装不装得上

这是 T6 唯一真正的前置问题，也是记忆里 D-9 卡住 47K 全量训练的那一个。

`make_link_loader` 有两条路：装了 `pyg-lib` 或 `torch-sparse` 就用 PyG 的
`LinkNeighborLoader`（C++ 采样），没装就退回纯 Python 的 `LightLinkLoader`。
T4 的 `pip freeze` 显示远端两个都没有，所以现在会走 Python 那条。
加 `--fanout` 之后 Python 那条的子图大小是有界的（T0 第 1 项就是干这个的），
**能跑，但每个 batch 的采样在 Python 里做**，100K 上可能慢到不可接受。

在计算节点上探一次：

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00
# 进入分配到的节点后
source ~/.mlf_env && source "$ENV_SOURCE/bin/activate"   # 或按 §3 的 staging 方式
python -c "import torch; print(torch.__version__)"
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.0+cu130.html
python -c "
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
print('pyg_lib', WITH_PYG_LIB, '| torch_sparse', WITH_TORCH_SPARSE)"
```

三种结果，对应三种走法：

| 探测结果 | 怎么走 |
|---|---|
| 装上了，两个标志有一个 True | 直接按 §5 跑。这同时解除 D-9，本身是一条独立结论，要写进报告 |
| 装不上（torch 2.12 / cu130 没有对应 wheel） | 还是按 §5 跑，但**第 3 步（100K 冒烟 2 epoch）的墙钟就是决定性的数据**：外推到 25 epoch 如果超过 4 小时，就得回头解决采样后端 |
| 装上了但 import 报错 | 别硬用，卸掉，按上一行走。半装的后端比没有更难查 |

把探测输出发我，我据此定 §5 第 3 步的 `--time`。

---

## 3. 环境与输入

环境沿用 T4 建好的那个（`ENV_SOURCE` + staging 到 `/tmp`，因为 `$HOME` 是 noexec）。
T6 不需要新包，除非 §2 决定装 `pyg-lib`。

需要在远端就位的文件：

| 文件 | 从哪来 | 备注 |
|---|---|---|
| `$DATA_ROOT/data1m/graphs/hgraph_100k.pt` | T5 的正式件 | 还没跑就先跑 T5（[`T5.md`](T5.md) §8） |
| `$DATA_ROOT/data1m/feats/100k/family_vocab.csv` | T4 | checkpoint 的绑定要它 |
| `$PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt` | 本地，**gitignore 里**，要单独 scp | 第 1 步 12K 锚点要用 |
| `$DATA_ROOT/data1m/graphs/hgraph_30k.pt` | 就是 `hgraph_ml_v2.pt`，改名或做软链 | 第 2 步要用 |

```powershell
scp D:\research\model_lake\codes\ModelLakeFishing\stage1BuildTransferGraph\hgraph_ml_v2_sub.pt `
    x98liu@watgpu.cs.uwaterloo.ca:<REMOTE_PROJECT_ROOT>/stage1BuildTransferGraph/
```

```bash
# 30K 那一档就是冻结的 CORE 图本身
ln -s "$PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
      "$DATA_ROOT/data1m/graphs/hgraph_30k.pt"
```

12K 那一档的图不在 `data1m/graphs/` 下，作业脚本按 `hgraph_<RUNG>.pt` 找，所以也做个软链：

```bash
ln -s "$PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt" \
      "$DATA_ROOT/data1m/graphs/hgraph_12k.pt"
```

---

## 4. 作业脚本

[`scripts/watgpu/train_rung.sbatch`](../../scripts/watgpu/train_rung.sbatch)，
请求 `--gres=gpu:1 --cpus-per-task=8 --mem=96G --time=04:00:00`，读三个环境变量
`RUNG` / `SEED` / `EPOCHS`。开的 T0 开关是 `--fanout --sparse-M --contrast-n-neg 256
--chunked-infer 50000`，和 T0 在 12K 上验收锚点时用的那一组相同。

`RUN_ID` 默认按 `R2_<RUNG>_s<SEED>_e<EPOCHS>` 生成。**作业被 preempt 重投时，
RUN_ID 保持不变**：驱动会自己找 `<run>/ckpt/last.pt` 续跑。换 RUN_ID 会把同一次
逻辑运行的指标劈成两个目录。

---

## 5. 四步，按顺序，不许跳

### 第 1 步：12K 锚点（§9.2 / G-C2）

```bash
source ~/.mlf_env
cd "$PROJECT_ROOT" && mkdir -p logs
RUNG=12k SEED=0 EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
  scripts/watgpu/train_rung.sbatch
```

**这一步是整条链路的验收，不是热身。** 它要回答的是：换了机器、换了 CUDA、
开了 T0 的四个开关之后，模型还是不是原来那个模型。

判据：训练本身只报 loss；`gold@10` 要等 T7 导出后才有。所以第 1 步的验收分两半——
现在能查的是机制门（`MANIFEST.json` 里 `mechanism_gate.passed == true`），
`gold@10 = 0.4159 ± 3-seed 噪声` 要等 T7/T8。**在那之前不要提交 100K 全量。**

### 第 2 步：30K 全 CORE

```bash
RUNG=30k SEED=0 EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
  scripts/watgpu/train_rung.sbatch
```

这是第一次跑通全部 30,183 个 CORE 模型（D-9 之前一直卡在这里，P3 只能子采样到 12K）。
跑通本身就是一条要写进报告的结果，跟 100K 无关。

### 第 3 步：100K 冒烟，2 epoch

```bash
RUNG=100k SEED=0 EPOCHS=2 sbatch --export=ALL,RUNG,SEED,EPOCHS \
  scripts/watgpu/train_rung.sbatch
```

**这一步的产出是三个数，不是模型**：单 epoch 墙钟、峰值显存、有没有 NaN。
`MANIFEST.json` 里的 `wallclock_s` 和 `peak_gpu_mem_gb` 就是第 4 步 `--time` 的依据。
把这两个数发我再提交全量。

### 第 4 步：100K 全量，三个 seed

```bash
for s in 0 1 2; do
  RUNG=100k SEED=$s EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
    scripts/watgpu/train_rung.sbatch
done
```

三个 seed 是为了给 A 轴的 `gold@10` 一个区间。单个 seed 的数字在这个项目里已经被
拦下过一次假信号（v4 W3 的 3-seed 结论在 8 seeds 下不成立），所以单 seed 的数不要拿来下结论。

---

## 6. 怎么判断跑对了

Slurm 报 `COMPLETED` 不等于通过。每个作业跑完看 `MANIFEST.json`：

| 字段 | 应该是什么 |
|---|---|
| `mechanism_gate.passed` | `true`。false 就停，把 `metrics/train_history.json` 发我 |
| `mechanism_gate.loss_descended` | `true` |
| `mechanism_gate.no_nan` | `true` |
| `n_models` | 12k→12000、30k→30183、100k→**100000** |
| `binding.graph_sha256` | 和 T5 报告里的 `graph_sha256` 一致 |
| `start_epoch` | 没被 preempt 就是 0；被 preempt 过是续跑点，这时 `resumed_from` 非空 |
| `peak_gpu_mem_gb` / `wallclock_s` | 记下来，第 4 步和 R3/R4 都靠它估 |

关于时间的一个提醒：本地 12K 是 29 秒/epoch、峰值 0.64 GB（RTX 4060）。
100K 不是简单乘 8.3——L1 的全湖 logQ 项每步都要对**全部**模型做一次前向，
这一项随 N 线性涨；但卡也换成了 48 GB 的 RTX 6000 Ada。
**所以我不给 100K 的预估，第 3 步实测出来是多少就是多少。** 计划里那个
「A100 上 25 epoch 约 30 分钟」是估的，而且卡都不是那张。

---

## 7. 跑完发我什么

checkpoint 很大（模型加 optimizer 状态），不用下载。每个 run 只要：

```bash
cd "${OUTPUT_ROOT}/runs/<RUN_ID>"
tar czf ~/t6_<RUN_ID>.tgz MANIFEST.json metrics/ metadata/ stdout/train.log
```

| # | 内容 | 用途 |
|---|---|---|
| 1 | `MANIFEST.json` | 机制门、墙钟、显存、binding、resume 情况 |
| 2 | `metrics/train_history.json` | 逐 epoch loss，看有没有该降不降 |
| 3 | `metadata/resolved_config.json` | 确认配置真的没被改过 |
| 4 | `stdout/train.log` + `logs/mlf-t6-train-<JOB_ID>.out` | 过程留痕 |
| 5 | `sacct -j <JOB_ID> --format=JobID,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode` | 排队时间、真实内存峰值 |

另外：§2 的 `pyg-lib` 探测输出。

---

## 8. 需要你定的两件事

### 问题 1：`--amp` 不做，可以吗

§9.1 列的开关里有 `--amp`，我没实现。理由：现在的 `train()` 里没有 autocast，
加进去会改变数值，而第 1 步的 12K 锚点正是要证明「换了环境、开了 T0 开关，模型还是原来那个」。
在同一批改动里既换环境又换精度，锚点偏了就没法归因。

而且 12K 峰值显存只有 0.64 GB，100K 在 48 GB 的卡上大概率也用不满，
**AMP 现在解决的不是一个我们有的问题**。建议留到 R3/R4 显存真的不够时再加，
并且单独跑一次锚点对拍。不同意就说，我现在加。

### 问题 2：12K 锚点没到 0.4159 怎么办

先说清楚会怎么处理，免得到时候临时决定：

| 偏差 | 处理 |
|---|---|
| 落在 3-seed 噪声内 | 通过，继续 |
| 明显偏低 | 逐个关 T0 开关重跑定位（`--fanout` / `--sparse-M` / `--contrast-n-neg`），**不改配置去凑** |
| 明显偏高 | 当成 bug 查，优先查划分和泄漏。这个项目在 P4 抓到过一次全图 `z_d` 让 held-out 查询看见自己标签、`gold@10` 从 0.42 虚高到 0.61 |

需要你确认的是最后一行的态度：**偏高不算好消息，一样停下来查。**

---

## 9. 容易踩的坑

| 症状 | 原因 | 处理 |
|---|---|---|
| 训练极慢，GPU 利用率很低 | 走了 Python 的 `LightLinkLoader`，采样成了瓶颈 | §2 的 `pyg-lib` 探测；短期先用 `--fanout` 压子图 |
| `RuntimeWarning: pyg-lib / torch-sparse not installed` | 同上，且没加 `--fanout` | 脚本里已经加了 `--fanout`；看到这条说明脚本被改过 |
| `IncompatibleCheckpoint: graph_sha256 ...` | run 目录复用了，但图换了 | 换个 `RUN_ID`，别删 checkpoint 硬续 |
| resume 后 loss 跳变 | RNG 没恢复 | §1.1 那个坑。确认跑的是修好之后的 commit |
| CUDA OOM | 扇出上限没生效或对比损失还是全 N² | 先把失败配置和峰值记下来，再减 `--batch-size`；**改了 batch size 要在报告里登记**，它是 L1L3b 配置的一部分 |
| CPU OOM / `MaxRSS` 撞 96G | 稀疏 M 没生效 | 确认 `--sparse-M` 在命令行里，查 `resolved_config.json` |
| 作业被 preempt 后指标重复 | 换了 `RUN_ID` | 同一逻辑运行始终用同一个 `RUN_ID`，驱动自己会续 |
| `AssertionError: graph has N models, rung ... expects` | 图和档位对不上 | 查软链指向；这条断言是计划 §1 第三条，故意在训练开始前就查 |

---

## 10. 这一步之后

T6 只产出 checkpoint 和 loss 曲线。`gold@10`、HNSW、冷启动分层都在 T7（导出加索引）
和 T8（三轴评测），那两步的代码还没写。第 1 步的锚点验收严格来说要等 T7 才能完成——
所以第 1 步跑完先别关，等我把 T7 写出来再一起验。

---

## 11. 无人值守的一次性工作流

[`scripts/watgpu/submit_t6_all.sh`](../../scripts/watgpu/submit_t6_all.sh) 把探测、训练、
机制门、证据归档和下一档提交串成一个状态机。提交器本身只提交两个短作业；每个验门作业
通过后才提交下一档，因此不会占着一个 CPU 作业等 GPU，也不会在失败后继续烧后续配额。

默认遵守本 runbook 的 gold gate，跑完 100K smoke 后暂停：

```bash
cd "$PROJECT_ROOT"
bash scripts/watgpu/submit_t6_all.sh
```

如果当前目标是先完成全部 T6 checkpoint，并明确接受「T7/T8 尚未验证 gold@10」，使用：

```bash
cd "$PROJECT_ROOT"
T6_ALLOW_NO_GOLD_GATE=1 bash scripts/watgpu/submit_t6_all.sh
```

这个 opt-in 会写入工作流的 `jobs.tsv`、每个 gate report 和最终 `SUMMARY.json`；它只允许
训练链继续，不会把锚点标成科学验收通过。自动链路为：

```text
pyg-lib 探测并选择持久环境
  -> 12K / 25 epoch -> 机制门
  -> 30K / 25 epoch -> 机制门
  -> 100K / 2 epoch -> 机制门 + 25 epoch 墙钟外推
  -> 100K / 25 epoch / seeds 0,1,2（并行）-> 各自机制门
  -> SUMMARY.json
```

任一机制门失败，后续作业不会被提交。smoke 外推超过四小时也会停止，符合 §2/§5 的
采样后端判据。作业状态在 `${OUTPUT_ROOT}/t6_workflows/<T6_WORKFLOW_ID>/`：

```bash
latest=$(ls -dt "${OUTPUT_ROOT}"/t6_workflows/T6_* | head -1)
cat "$latest/STATUS"
cat "$latest/jobs.tsv"
cat "$latest/SUMMARY.json" 2>/dev/null || true
```

`STATUS=TRAINING_COMPLETE` 只表示 T6 训练和机制门完成；完整锚点仍须 T7/T8。

---

## 12. 2026-08-11 正式执行结果

工作流 `T6_20260811T174245Z` 已完成，最终 `STATUS=TRAINING_COMPLETE`、
`SUMMARY.json: passed=true`，所有训练与验门作业均为 Slurm `COMPLETED / ExitCode 0:0`。

环境探测在计算节点安装并验证了 `pyg-lib 0.8.0+pt212cu130`，
`torch_geometric.typing.WITH_PYG_LIB=true`，因此 D-9 的 Python 采样后端风险解除。
实际训练固定在预检通过的 `watgpu808`（NVIDIA H200 NVL，143,771 MiB）；
环境从 noexec NFS 自动 staging 到容量与执行权限均通过检查的节点本地文件系统。

| run | N | epochs | loss first → last | wallclock | peak GPU | 机制门 |
|---|---:|---:|---:|---:|---:|---|
| `R2_12k_s0_e25` | 12,000 | 25 | 19.0257 → 13.8466 | 93.0 s | 0.583 GB | PASS |
| `R2_30k_s0_e25` | 30,183 | 25 | 18.9898 → 15.0575 | 105.5 s | 0.866 GB | PASS |
| `R2_100k_s0_e2` | 100,000 | 2 | 19.8590 → 18.2148 | 14.2 s | 1.719 GB | PASS |
| `R2_100k_s0_e25` | 100,000 | 25 | 19.6944 → 15.4304 | 108.5 s | 1.737 GB | PASS |
| `R2_100k_s1_e25` | 100,000 | 25 | 19.3075 → 14.8090 | 116.6 s | 1.713 GB | PASS |
| `R2_100k_s2_e25` | 100,000 | 25 | 19.8016 → 14.8894 | 106.7 s | 1.723 GB | PASS |

三档正式图 SHA-256：

```text
12k  00ac2434cd8ecaa2f633bc377b25d1d5ae0a7d50cbcab88774355efc47a11d34
30k  e6ae2dfeb59779f4cb0242a08bf90d6628699e90a9a648048b799d9cc7d71cda
100k 7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0
```

六个 run 的 `last.pt` / `best.pt` 均保留在远端 `${OUTPUT_ROOT}/runs/<RUN_ID>/ckpt/`；
100K 三个完整 seed 的 checkpoint 均绑定并保存了同一份 `family_vocab.csv`。
按 §7 不下载 checkpoint，证据包已校验并回收到本地：

```text
D:\research\model_lake\data\runs\T6_20260811T174245Z\
```

本地复验包括：delivery SHA-256、六个 artifact SHA-256、六份 MANIFEST/metrics 合约、
图 binding、配置开关、epoch/history 长度与全作业 `sacct`。全部通过。

**尚未完成的不是 T6 训练，而是 G-C2 科学锚点。** `gold@10=0.4159 ± 噪声`
必须等 T7 held-out 导出与 T8 评测，当前 `gold10_validated=false` 保持正确。
