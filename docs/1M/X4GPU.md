# X4：在 watGPU 上重训（混合提议分布与 `global_n_datasets`）

上级：[`GOLD10_FAST_LEVERS_AUDIT.md`](GOLD10_FAST_LEVERS_AUDIT.md) §8 顺序 4。
诊断来自 [`X1.md`](X1.md) §4 与 §5，基线来自 [`F6.md`](F6.md) 与 [`F8.md`](F8.md)。

本文写给执行者。范围是从改代码开始，到把 checkpoint 与运行记录交回本地为止。
导出、四轴评测与 A/B 结论写在本地做，列在 §12，不在远端范围内。

先读 §2 到 §5 再动手。§2 是预注册，动手之后不能改；§3 说明哪些量必须保持与 F6 一致，
否则这次的数字不能与 F8 并排；§4 是五处代码缺口；§5 是规模上的代价估计。

## 1. 目标与交付

X1 §4 与 §5 量到两件事，都是配置值而不是数据或模型的性质。

第一，`lake_alpha=0.75` 让提议分布 `q` 只有 3.52% 的质量落在训练可见的带监督模型上
（三 seed 0.0355 / 0.0369 / 0.0332），256 个负样本里平均只有 9.0 个是能真正挤占 gold 的模型。
100K 档同一口径是 61.38% 与 157.1 个，差 17.4 倍。

第二，`global_n_datasets=16` 让每个训练可见的查询数据集在 25 个 epoch 里
平均只被全局项算到 1.47 次（三 seed 1.51 / 1.45 / 1.44），100K 档是 4.06 次，差 2.77 倍。

两者相乘是 48 倍。这是相关而非因果：三档梯子的监督集与查询集本来就不同，
所以 X1 只能说这是两条可测、可控的候选原因。判断因果需要把这两个值改掉再训一次，
那就是本阶段。

本阶段产出 checkpoint、loss 曲线与运行记录，不产出精度数字。
`gold@10` 要等本地导出之后才有（§12）。远端能验收的只有机制门与配置门（§9）。

九个 run（三条实验臂 × 三个 split seed），每个交回一个约 12 MB 的 checkpoint 加四份小文件（§11）。

## 2. 预注册

这一节在提交任何作业之前写定，执行期间不修改。看到数字之后再改判据，
这次比较就不能作为因果证据。

主指标是 A 轴 `gold@10`：候选池为全部 3,016,439 个模型，嵌入只用 held-out 的 `z_*_eval`，
查询集由 `split_seed` 决定因而与 F8 完全相同，三个 seed 取均值。
基线是 F8 §2.1 的 0.0598（逐 seed 0.0700 / 0.0399 / 0.0696）。

判据：合并臂（同时改两处）的三 seed 均值 `gold@10` 达到 0.09 或以上，
并且三个 seed 各自都高于自己的 F6 对应值，则认为 X1 §4 与 §5 的两条诊断得到因果支持。
均值低于 0.075，或三个 seed 的变化方向不一致，则认为这两个被冻结的比例型超参
不是 100K 到全量档差距的主要原因，X1 的那两条仍然只是相关。
落在 0.075 与 0.09 之间的情况按「方向对但幅度不足」报告，不改判据。

同时报告、不设阈值的次要量：`gold@1`、`top3@10`、`gold-gap@10`、`root_gold@10`、
中位 gold 排名，以及 C 轴的分层几何。C 轴不预期改善——
塌缩是输入特征的问题（brainstorm A 组），本阶段没有动特征。

三条实验臂，各三个 seed：

| 臂 | 改动 | 用途 |
|---|---|---|
| G | `--lake-gamma 0.5` | 只改提议分布 |
| D | `--global-n-datasets 128` | 只改每步覆盖的数据集数 |
| GD | 两者同时 | 审计 §8 顺序 4 要的那一条 |

F6 的三个 run 就是第四臂（两处都不改），已经存在，不重跑。
先跑 GD，它是要回答的问题；G 与 D 用来分开两条诊断各自的贡献，
合并臂如果动了而两个单改臂都不动，说明是交互作用而不是任一单项。

`gamma = 0.5` 的取值来自 X1 §4：它把 256 个负样本里带监督的期望个数带回 132.5，
与 100K 档的 157.1 同一量级。与 100K 完全对齐需要 `gamma ≈ 0.60`，
本次不取那个值，因为 0.5 是审计里写下的数，改动越少越容易解释。

`global_n_datasets = 128` 同样来自审计。它把每个数据集的触达次数带到约 11.7 次，
比 100K 档的 4.06 次高 2.9 倍，也就是说这个值不是与 100K 对齐，而是越过了对齐点
（对齐大约在 44）。这一点在报结果时要写明，不要把 128 说成「恢复到 100K 的水平」。

## 3. 必须与 F6 保持一致的量

除了 §2 那两个键，`l1l3b_config()` 的其余键一个都不动。
`metadata/resolved_config.json` 会记下实际生效的配置，跑完与 F6 的对应文件逐键比对，
差异应当恰好是 `lake_gamma` 与 `global_n_datasets` 两项。

图必须是同一张。三个 F6 run 绑定的摘要是
`0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`，
本地重算一致。远端 `hgraph_rf/` 已在 F6 期间上传并逐分片核验过，
其中 `meta.json` 是 F6 修正过的那一份，SHA-256 为
`4ed7b5d13395d3e5fe98faa3e3b546199e6c569687e2b0d27fbd278676398fa0`。
本阶段不重传图，只确认摘要（§6）。

`init_seed` 恒为 0，`--seed` 传的仍是 `split_seed`。三个 seed 给的是划分噪声区间。

特征保持 float32，不开 `--amp`。九个 run 用同一个 git commit，
不要在中途改代码再跑剩下的臂。

一件需要提前知道的事：加了 `gamma` 之后 logQ 修正项的分母变了，
所以 loss 的绝对值不再与 F6 的 22.29 → 16.63 可比。机制门只看是否下降与有无 NaN，
仍然有效；但不要把不同的 loss 数值当成异常。
`global_n_datasets` 不影响 loss 的量纲，因为 `global_lake_loss` 对数据集取的是平均而非求和，
改的只是这个平均的方差。

## 4. 代码缺口：五处

前四处在远端跑之前要做完，第五处是本地评测时才用到的。
改完在本地跑一遍 `pytest scale1m/tests` 与 `pytest stage2TrainGraphSAGE/tests`，
F6 时的基线是 298 passed，数字不应该变小。

### 4.1 `build_lake_logq` 加混合成分

[`stage2TrainGraphSAGE/losses.py:1015`](../../stage2TrainGraphSAGE/losses.py#L1015)
现在只产生幂律提议分布。加一个默认关闭的第二成分：

```python
def build_lake_logq(trained_on_index, num_models, *, alpha=0.75, n0=1.0, gamma=0.0):
    """...

    gamma > 0 混入一个在带监督模型上均匀的成分：

        q = (1-gamma) * (deg+n0)^alpha / Z  +  gamma * [deg>0] / n_labeled

    采样与 -log q 修正都用这一个混合密度，所以 sampled softmax 的分母仍然是全湖
    softmax 的无偏估计。无标签模型在第一成分里仍有正质量，q.log() 处处有限。
    gamma=0.0 时不进入这个分支，返回值与改动前逐元素相同。
    """
    deg = torch.bincount(trained_on_index[0], minlength=num_models).float()
    w = (deg + float(n0)) ** float(alpha)
    q = w / w.sum()
    if gamma > 0.0:
        lab = (deg > 0).to(q.dtype)
        n_lab = lab.sum()
        if n_lab > 0:
            q = (1.0 - float(gamma)) * q + float(gamma) * lab / n_lab
    return q, q.log()
```

配一条测试：`gamma=0.0` 时与改动前 `torch.equal`；
`gamma=0.5` 时 `q.sum()` 仍为 1，且 `q[deg > 0].sum()` 等于
`0.5 + 0.5 * q_old[deg > 0].sum()`。

### 4.2 `ablation.py` 把 gamma 传下去，并把质量打进日志

[`stage2TrainGraphSAGE/ablation.py:191`](../../stage2TrainGraphSAGE/ablation.py#L191)
那次调用加一个参数，同时把已有的那行打印补上两个可核对的量：

```python
                q, logq = build_lake_logq(
                    ti, data["model"].num_nodes,
                    alpha=cfg.get("lake_alpha", 0.75), n0=cfg.get("lake_n0", 1.0),
                    gamma=cfg.get("lake_gamma", 0.0))
                deg = torch.bincount(ti[0], minlength=data["model"].num_nodes)
                mass = float(q[deg > 0].sum())
                print(f"    [lake logQ] models={q.numel()} labeled={(deg > 0).sum().item()} "
                      f"max_deg={int(deg.max())} q_head={q.max():.4f} "
                      f"alpha={cfg.get('lake_alpha', 0.75)} "
                      f"gamma={cfg.get('lake_gamma', 0.0)} "
                      f"mass_on_labeled={mass:.4f} "
                      f"E_labeled_in_negs={mass * cfg.get('global_n_neg', 64):.1f}")
```

这一行是 §9 最有用的一道门：`mass_on_labeled` 的值可以事先算出来，
与 X1 §4 的表逐 seed 对照。

### 4.3 `train_rung.py` 两个开关

[`scale1m/train_rung.py:210`](../../scale1m/train_rung.py#L210) 之后加两个参数，
默认 `None`，所以 F6 那条命令行解析出的配置一字不变：

```python
    p.add_argument("--lake-gamma", type=float, default=None,
                   help="mixture weight of the uniform-over-labeled component in q")
    p.add_argument("--global-n-datasets", type=int, default=None,
                   help="datasets scored by the global term each step (default 16)")
```

[`build_config`](../../scale1m/train_rung.py#L176) 里按条件写入，不要放进那个无条件的
`cfg.update(...)`：

```python
    if args.lake_gamma is not None:
        cfg["lake_gamma"] = args.lake_gamma
    if args.global_n_datasets is not None:
        cfg["global_n_datasets"] = args.global_n_datasets
```

`ablation.py` 早就在读 `cfg.get("global_n_datasets", 16)`，所以这个键不需要别的改动。

### 4.4 新的作业脚本

新写 `scripts/watgpu/train_rung_x4.sbatch`，不要改 `train_rung_rf.sbatch`——
那份脚本还要能原样复现 F6 的三个 run。新脚本在它的基础上加三件事：
从环境变量读 `ARM`，据此拼出 `--lake-gamma` / `--global-n-datasets`，
以及把 `RUN_ID` 的默认值改成 `X4${ARM}_full_s${SEED}_e${EPOCHS}`。

```bash
: "${ARM:?set ARM to G, D or GD}"
case "${ARM}" in
  G)  EXTRA=(--lake-gamma 0.5) ;;
  D)  EXTRA=(--global-n-datasets 128) ;;
  GD) EXTRA=(--lake-gamma 0.5 --global-n-datasets 128) ;;
  *)  echo "[fail] unsupported ARM: ${ARM}" >&2; exit 2 ;;
esac
: "${RUN_ID:=X4${ARM}_full_s${SEED}_e${EPOCHS}}"
```

其余部分（节点、内存、环境 staging、`--fanout --sparse-M --contrast-n-neg 256
--chunked-infer 50000 --skip-diagnostics`）与 `train_rung_rf.sbatch` 相同，
`"${EXTRA[@]}"` 加在 `srun python -m scale1m.train_rung` 的参数末尾。

### 4.5 本地评测入口的 `--run-fmt`

[`scale1m/eval_rf.py:58`](../../scale1m/eval_rf.py#L58) 与
[`scale1m/fast_lever_audit.py:66`](../../scale1m/fast_lever_audit.py#L66)
都把 `RUN_FMT = "RF_full_s%d_e25"` 写死了，X4 的导出目录名不同，评测会找不到。
两个文件各加一个 `--run-fmt` 参数，默认值保持现在这一串，并把导出目录的模块级常量用处改成
读 `args`。checkpoint 回来后又补齐了两处边界：`eval_rf.py` 的 `--full-sidecar` 默认值在解析
参数后由实际 `--exports` / `--run-fmt` 派生；D 轴的训练 MANIFEST 使用独立的
`--f6-runs` / `--f6-run-fmt`，缺少任一 seed 时直接报错，不再静默写出空的 `train_per_seed`。

## 5. 规模上的代价

两处改动对显存的影响都很小，对墙钟的影响需要冒烟来定。

混合提议分布只在建 `q` 时多做一次长度为 3,016,439 的向量运算，一次性，可忽略。
采样与修正的成本与改动前相同：`torch.multinomial` 的调用次数、每次的样本数都没变。

`global_n_datasets` 从 16 到 128 把
[`global_lake_loss`](../../stage2TrainGraphSAGE/losses.py#L1036)
的内层 Python 循环从 16 次变成 128 次。每次循环做一次 3M 长度分布上的
`torch.multinomial`（256 个样本）、一次 `torch.isin`、两次小矩阵乘和一次 `logsumexp`，
再加约十次 CUDA kernel 启动。

按 F6 的实测折算，seed 0 的 25 个 epoch 是 426.4 秒，每 epoch 51 步，
即每步约 334 毫秒。F6GPU §5.1 已经确认每步的开销由那次带梯度的全图前向主导
（3,016,439 个节点、`x` 是 5.41 GB）。全局项目前占每步的比例没有单独测过，
所以这里只能给一个区间：如果它现在占 3%，128 之后每步约 1.2 倍；
如果占 10%，约 1.7 倍。冒烟会给出真值。

显存方面，全局项每个数据集的临时量是 256 × 128 的量级，128 个数据集加起来仍然远小于
两份 `x`。预计峰值仍在 38 GB 附近（F6 三个 run 是 38.001 / 38.015 / 37.976 GB），
冒烟确认。

配额估计：单个 run 按 F6 的 379–499 秒加上上面的倍率，取 1.7 倍上界约 14 分钟，
九个 run 合计约 2.1 GPU-hours。watgpu508 有三张 H200，同一条臂的三个 seed 可以并行。

## 6. 远端已有什么

本阶段不需要传数据。F6 期间上传的这些应当还在，先确认再提交作业：

```bash
source ~/.mlf_env
ls -l "$DATA_ROOT/data1m/graphs/hgraph_rf" "$DATA_ROOT/data1m/feats_rf/family_vocab.csv"
sha256sum "$DATA_ROOT/data1m/graphs/hgraph_rf/meta.json" \
          "$DATA_ROOT/data1m/feats_rf/family_vocab.csv"
```

应当是：

```text
4ed7b5d13395d3e5fe98faa3e3b546199e6c569687e2b0d27fbd278676398fa0  meta.json
00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005  family_vocab.csv
```

`meta.json` 不是 F6GPU §6 预写的 `82cb04…`，这是 F6 期间修正 `GRAPH_REPORT.json` 条目
留下的已知偏离，不要「修回去」。八个分片的逐一复算脚本在 F6GPU §6，
如果这次要重跑一遍，`graph_digest` 应当仍是 `0e80b839…`。

如果图不在了，按 F6GPU §6 重传，并在回报里写明重传过。

代码的同步方式需要注意：F6 期间发现远端 `origin` 是停在 `7d3457a…` 的旧 bundle，
`git pull` 取不到当前分支。当时的做法是定向同步改动过的文件并逐个记 SHA-256。
本阶段沿用同样的做法，需要同步的是 §4.1 到 §4.4 的四个文件：

```text
stage2TrainGraphSAGE/losses.py
stage2TrainGraphSAGE/ablation.py
scale1m/train_rung.py
scripts/watgpu/train_rung_x4.sbatch
```

同步后在远端逐个 `sha256sum` 并写进回报，同时对四个 Python 文件跑 `py_compile`、
对 sbatch 跑 `bash -n`。

## 7. 环境前置

环境沿用 F6 用的那份。F6 期间在 `$ENV_SOURCE` 里装上了
`pyg_lib 0.8.0+pt212cu130` 与 `matplotlib 3.11.1`，如果那次的安装被持久化，
本阶段不需要再装。先确认一次：

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 --nodelist=watgpu508
source ~/.mlf_env && source "$ENV_SOURCE/bin/activate"
python -c "
import torch, matplotlib
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
print('torch', torch.__version__, '| pyg_lib', WITH_PYG_LIB, '| torch_sparse', WITH_TORCH_SPARSE)"
```

两个标志都为 False 时 `make_link_loader` 会退回纯 Python 的 `LightLinkLoader`，
在 3M 上会慢到不能接受，按 F6 的做法在计算节点上补装并回报探测输出。

## 8. 执行顺序

### 8.1 本地自检

在同步任何东西到远端之前，本地跑一次小图上的等价性检查，确认 `gamma=0.0` 没有改变行为：

```bash
cd d:/research/model_lake/codes/ModelLakeFishing
./.venv/Scripts/python.exe -m pytest scale1m/tests stage2TrainGraphSAGE/tests -q
```

再在本地 CPU 上用 12K 图跑一个 1 epoch 的 run（`--rung 12k`，不传两个新开关），
确认 `metadata/resolved_config.json` 里没有出现 `lake_gamma` 与 `global_n_datasets`。
这一步保证 F6 的配置没有被这次改动带偏。

### 8.2 一个 epoch 的冒烟，只跑 GD

```bash
cd "$PROJECT_ROOT" && mkdir -p logs
ARM=GD RUNG=full SEED=0 EPOCHS=1 RUN_ID=X4GD_full_s0_smoke \
  sbatch --time=00:30:00 --export=ALL,ARM,RUNG,SEED,EPOCHS,RUN_ID \
  scripts/watgpu/train_rung_x4.sbatch
```

这一步要回报四个数：单 epoch 墙钟、`peak_gpu_mem_gb`、`sacct` 的 `MaxRSS`，
以及 `stdout/train.log` 里 `[lake logQ]` 那一行的 `mass_on_labeled`。
第四个数应当是 0.5177（seed 0），偏差超过 0.001 就先停下来查 §4.1 的实现。

F6 的同规格冒烟是 37.1 秒、37.991 GB，可以直接对比。
`--time` 与 `--mem` 从这里定；不要留大额占位值，Slurm 按申请的墙钟排队。

### 8.3 三条臂的正式训练

GD 先跑，三个 seed 并行：

```bash
for s in 0 1 2; do
  ARM=GD RUNG=full SEED=$s EPOCHS=25 \
    sbatch --time=<按 8.2 定> --export=ALL,ARM,RUNG,SEED,EPOCHS \
    scripts/watgpu/train_rung_x4.sbatch
done
```

三个都 `COMPLETED` 且通过 §9 之后再提交 G 与 D，命令相同，只改 `ARM`。
分两批而不是九个一起提交，是为了在 GD 出问题时不浪费另外六次排队。

作业被 preempt 重投时保持同一个 `RUN_ID`，驱动会自己从 `<run>/ckpt/last.pt` 续跑。

## 9. 怎么判断跑对了

Slurm 报 `COMPLETED` 不等于通过。每个 run 看 `MANIFEST.json` 与 `stdout/train.log`：

| 字段 | 应该是什么 |
|---|---|
| `mechanism_gate.passed` | `true`；false 就停下并把 `metrics/train_history.json` 一起回报 |
| `mechanism_gate.no_nan` | `true` |
| `n_models` | 3016439 |
| `binding.graph_sha256` | `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`，九个 run 相同，且与 F6 相同 |
| `binding.family_vocab_sha256` | `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005` |
| `binding.num_families` | 41056 |
| `init_seed` | 0 |
| `seed` | 0 / 1 / 2 |
| `metadata/resolved_config.json` 的 `resolved_config` | 与 F6 对应文件逐键比对，差异恰好是本臂改的那一到两个键 |
| `metadata/resolved_config.json` 的 `git_head` | 非空，且九个 run 相同 |
| `peak_gpu_mem_gb` / `wallclock_s` | 记下来 |

`[lake logQ]` 那一行的 `mass_on_labeled` 按臂与 seed 对照下表。
G 与 GD 用 gamma=0.5 的一列，D 用 gamma=0 的一列（它应当与 F6 相同）。

| seed | 训练可见带监督模型 | gamma=0 的质量 | gamma=0.5 的质量 | gamma=0.5 时 256 个负样本里的期望个数 |
|---:|---:|---:|---:|---:|
| 0 | 37,755 | 0.0355 | 0.5177 | 132.5 |
| 1 | 39,930 | 0.0369 | 0.5184 | 132.7 |
| 2 | 35,991 | 0.0332 | 0.5166 | 132.2 |

`labeled=` 一列也应当与表里的第二列逐 seed 相同，它是 F6 日志里已经出现过的量。

## 10. 已知的坑

| 症状 | 原因 | 处理 |
|---|---|---|
| `mass_on_labeled` 是 0.0355 而命令行传了 `--lake-gamma 0.5` | §4.3 的两行没有写进 `build_config`，或者写进了那个无条件的 `cfg.update` 之后被覆盖 | 看 `metadata/resolved_config.json` 里有没有 `lake_gamma` 这个键 |
| `q` 里出现 `-inf` 或 loss 变 NaN | 混合成分写成了「只在带监督模型上采样」，无标签模型的 `q` 变成 0 | §4.1 的写法保留了第一成分，无标签模型仍有正质量 |
| loss 的绝对值与 F6 差很多 | gamma 改变了 logQ 修正的分母 | 这是预期的，见 §3 最后一段；只看是否下降 |
| 墙钟远超冒烟的外推 | `global_n_datasets=128` 的 Python 内层循环在这台卡上比预计更贵 | 记下来照实报；不要为了追平时间去改 `n_neg` 或别的键 |
| `argument --lake-gamma: ...` 之类的解析错误 | 远端同步的是旧版 `train_rung.py` | 按 §6 逐个核对四个文件的 SHA-256 |
| `IncompatibleCheckpoint: graph_sha256 ...` | run 目录复用了但图或摘要变了 | 换一个 `RUN_ID`，不要删 checkpoint 硬续 |
| 训练跑完后长时间挂住 | 漏了 `--skip-diagnostics` | 见 F6GPU §5.3 |
| CPU 内存撞上 `--mem` | `train_data.clone()` 会把 memmap 的 5.41 GB 实体化 | `--mem` 保持 128G；`sacct` 的 `MaxRSS` 是真实峰值 |

## 11. 跑完交回什么

checkpoint 这次要下载，因为导出与评测在本地做（F7 就是这样做的，见 F7.md §7）。
每个 run 打一个包，只带 `last.pt`，不要带中间 epoch 的 checkpoint：

```bash
cd "${OUTPUT_ROOT}/runs/<RUN_ID>"
tar czf ~/x4_<RUN_ID>.tgz \
  MANIFEST.json metrics/ metadata/ stdout/train.log \
  ckpt/last.pt ckpt/family_vocab.csv
```

单个包约 12 MB，九个合计约 110 MB。

| # | 内容 | 用途 |
|---|---|---|
| 1 | `MANIFEST.json` | 机制门、墙钟、显存、binding |
| 2 | `metrics/train_history.json` | 逐 epoch loss |
| 3 | `metadata/resolved_config.json` | §9 的配置门 |
| 4 | `stdout/train.log` 与 `logs/mlf-*-<JOB_ID>.out` | `[lake logQ]` 那一行与过程留痕 |
| 5 | `ckpt/last.pt` 与 `ckpt/family_vocab.csv` | 本地导出要用 |
| 6 | `sacct -j <JOB_ID> --format=JobID,JobName,State,NodeList,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode` | 排队时间与真实内存峰值 |

另外回报：§6 的四个文件 SHA-256、§7 的采样后端探测输出、§8.2 冒烟的四个数。

## 12. 回到本地之后做什么

这一段不在远端范围内，写在这里是为了让执行者知道交回的东西会被怎么用。

把九个包解到 `docs/1M/X4_runs/<RUN_ID>/`，布局与 `docs/1M/F6_runs/` 相同。
然后每个 run 跑导出的前两个 stage：

```bash
python -m scale1m.export_rf --run docs/1M/X4_runs/<RUN_ID> --stage embed \
    --out <DATA>/data1m/exports_x4/<RUN_ID>
python -m scale1m.export_rf --run docs/1M/X4_runs/<RUN_ID> --stage metrics \
    --out <DATA>/data1m/exports_x4/<RUN_ID>
```

`index` 与 `curve` 两个 stage 这次不跑。A 轴用的是全扫式的流式打分，不需要 HNSW；
索引只有 B 轴要用，而 B 轴测的是检索结构而不是这次改的配置。
跳过它们省掉每个 run 约 2.2 GB 的索引和十二个子索引。
即便如此，`z_m.npy` 与 `z_m_eval.npy` 各 1.54 GB，九个 run 约 28 GB，落盘前先确认磁盘。

接着按 §4.5 加好 `--run-fmt` 之后跑评测：

```bash
python -m scale1m.eval_rf --axis a --exports <DATA>/data1m/exports_x4 \
    --run-fmt "X4GD_full_s%d_e25"
```

主指标由这一步给出，与 F8 §2.1 并排。之后按需要补 P 轴与 S 轴：
两者都要先按 X2 的方式为新的导出目录建 sidecar（sidecar 的内容只由图与 split seed 决定，
但 sidecar 的行序检查绑定的是导出目录的 id 快照，所以每个导出目录建一次）。
S 轴未显式传 `--full-sidecar` 时，现在会自动使用
`<exports>/<run-fmt % 0>/prior_sidecar.npz`。若之后补 D 轴并希望报告 X4 而非 F6 的训练成本，
训练 run 的根目录与格式必须独立传入：

```bash
python -m scale1m.eval_rf --axis d --exports <DATA>/data1m/exports_x4 \
    --run-fmt "X4GD_full_s%d_e25" --f6-runs docs/1M/X4_runs \
    --f6-run-fmt "X4GD_full_s%d_e25"
```

D 轴现在会在任一训练 MANIFEST 缺失时直接失败，不再静默留下空的 `train_per_seed`。

结论写进 `docs/1M/X4.md`，格式与 X1 到 X3 一致，并按 §2 的判据逐条对照预注册。

## 13. 需要测量并回报的判断点

这四条由执行者测量并回报，不要自行定案。

冒烟的单 epoch 墙钟、峰值显存、`MaxRSS`，以及 `mass_on_labeled`。前三个决定 §8.3 的
`--time` 与 `--mem`，第四个是 §4.1 实现是否正确的直接证据。

`global_n_datasets=128` 实际带来的每步开销倍率。§5 只能给 1.2 到 1.7 的区间。
如果实测超过 2 倍，回报之后再决定是否需要把内层循环向量化，
本阶段不要顺手改——那会让九个 run 不在同一份代码上。

远端 `hgraph_rf/` 是否还在、摘要是否仍是 `0e80b839…`。不在就要重传，
重传要写进回报，因为它会改变「这次与 F6 用的是同一张图」这个前提的证据链。

G 与 D 两条单改臂是否要跑。它们不是审计要的那一条，但合并臂如果动了，
没有这两条就无法区分是哪一处在起作用。配额紧张时可以只跑 GD 并说明，
但要在报告里写清楚这次没有区分两个因素。

---

## Codex 执行反馈

> 本节由执行者维护，用来区分操作指南与本次实际执行记录。

状态：**代码、自检、远端数据门与定向同步已完成；环境探针和 GD 冒烟已提交，正在等待
watgpu508 的 GPU；正式训练尚未提交。**

### 1. 本地实现与自检（2026-08-27）

- 已按 §4.1--§4.4 实现混合 `q`、配置透传、日志质量项和独立 X4 sbatch；另按 §4.5
  为 `eval_rf.py` 与 `fast_lever_audit.py` 增加了默认保持兼容的 `--run-fmt`。
- 新增测试同时检查 `gamma=0.0` 的逐元素等价性，以及 `gamma=0.5` 的归一化、带监督质量公式
  和有限 `logq`。完整回归结果为 **322 passed**（不低于 F6 的 298 基线）。
- 本地 CPU 12K / seed 0 / 1 epoch 等价性 run 写在 `docs/1M/X4_local_12k_equiv/`：
  训练墙钟 77.3 秒、无 NaN；单 epoch 因首末 loss 相同而按预期返回 mechanism gate false。
  其 `resolved_config` 与 F6 seed 0 的配置逐键比较为 **0 项差异**，且没有 `lake_gamma` 或
  `global_n_datasets` 键。

### 2. 远端输入与同步门

- SSH 公钥可无交互登录 `x98liu@watgpu.cs.uwaterloo.ca`；没有读取或要求任何秘密。
- 远端 `hgraph_rf/` 和 `family_vocab.csv` 均仍存在，没有重传。摘要分别命中：
  `meta.json = 4ed7b5d13395d3e5fe98faa3e3b546199e6c569687e2b0d27fbd278676398fa0`，
  `family_vocab.csv = 00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`。
- 按 §6 定向同步了四个文件；远端逐项 SHA-256 为：

| 文件 | SHA-256 |
|---|---|
| `stage2TrainGraphSAGE/losses.py` | `c9b35119ac64471352dbc75da372a3b065d60f8b630b3954ed45c94711392b9c` |
| `stage2TrainGraphSAGE/ablation.py` | `e0fa475bbdf2cc1040c9643b37a28d18d54f7cf08fd6b73f1af7650552a9191d` |
| `scale1m/train_rung.py` | `08904b015de386188b452b3a99f3b598207595460d605dd9452b6f4d05743a46` |
| `scripts/watgpu/train_rung_x4.sbatch` | `f1237a73f7ea6ccccc73fe68897297b51cfe69d2ce6bfb4d5d5af02cd4b67e9b` |
| `scripts/watgpu/x4_env_probe.sbatch` | `9c91b71de677ae89a309b25e2f6dba535da47a6e3d746f63db64991cbf4f9b24` |

三份 Python 文件的远端 `py_compile` 和 sbatch 的 `bash -n` 均通过。
`train_rung_x4.sbatch` 首次同步时的摘要确为
`a0b2ab221df7593f364bc4f36d74fce3b358b4a7a5cf939bad9b2491b4c79986`；§6 为绕过 watgpu608
故障的 `srun` job-step 增加 `X4_DIRECT=1` 后，正式执行版本变为表中的 `f1237a73...`。
交付物 `x4_execution/code_sha256.txt` 与本地实际文件均记录并命中最终摘要。

### 3. 当前 Slurm 队列

| 阶段 | Job ID | 状态 | 依赖 / 资源 |
|---|---:|---|---|
| H200/PyG 环境探针 | `1522221` | `PENDING (Priority)` | 任意可用 H200（排除 1103），1 GPU / 8G / 15m |
| GD / seed 0 / 1 epoch 冒烟 | `1522222` | `PENDING (Dependency)` | `afterok:1522221`，任意可用 H200 / 128G / 30m |

提交时 watgpu508 的 7 张 GPU 全部已分配；因此尚无 §8.2 的四个实测数，也没有提交任何
25 epoch 作业。冒烟只有在环境探针成功后才会启动；正式 GD 三 seed 仍须等冒烟产物通过
§8.2/§9 后提交。

### 4. H200 调度池扩展（2026-08-27）

最初固定到 watgpu508 的 `1522037` / `1522038` 始终未启动。执行者随后检查了 watGPU 的全部
H200 节点（508、708、808、908、1103、1208、1209），并把 X4 脚本从固定节点改为
`--constraint=H200`，让 Slurm 选择第一台可用的 H200。

该策略立即把探针调度到 watgpu1103，但探针确认 1103 没有足够大的可执行本地暂存文件系统来
复制持久化环境；正式训练也会在同一处失败，因此 1103 被显式排除。两次 1103 探测均在加载图
之前退出，没有训练产物。当前 `1522221` / `1522222` 会在其余六台 H200 中择机运行；此前所有
被替换作业在取消时均仍为 PENDING，没有重复执行或删除任何 run。

### 5. 非 H200 备选与正式 GD 启动（2026-08-27）

全站 H200 的预计最早启动一度为 2026-08-28 20:47，因此又检查了 Blackwell、L40S 与
ADA6000。Blackwell 节点或处于 DRAIN，或没有足够主存；ADA6000 环境探针 `1522226` 虽完成，
但报告 `cuda=False`，其依赖 smoke 在启动前取消。L40S 环境探针 `1522224` 在 watgpu408
完成，报告 `torch 2.12.0+cu130`、`cuda=True`、`pyg_lib=True`。

独立的 L40S 冒烟 `1522225` 使用 `RUN_ID=X4GD_full_s0_smoke_l40s` 完整跑完 1 epoch；Slurm
状态为 FAILED 仅因为单 epoch 的 `loss_descended=false` 使程序按约定返回 1。实测为：

| 墙钟 | 峰值 GPU | MaxRSS（python step） | `mass_on_labeled` | NaN |
|---:|---:|---:|---:|---|
| 170.1 s | 37.941 GB | 34,409,792 K | 0.5177 | 无 |

`E_labeled_in_negs=132.5`，与预注册表一致。L40S 的单 epoch 墙钟约为 F6/H200 冒烟的 4.6 倍，
但 48GB 显存能够容纳本实验，且 96G CPU 内存有充分余量。执行者据此取消仍未启动的 H200
候选链，提交正式 GD 三 seed 到共享 L40S 池，时限按实测设为 1h30m：

| seed | Job ID | 当前状态 |
|---:|---:|---|
| 0 | `1522230` | PENDING（watgpu408 上跑到 epoch 4 后被抢占，`Restarts=1`） |
| 1 | `1522231` | PENDING（预计 2026-08-27 22:13 在 watgpu408 启动） |
| 2 | `1522232` | PENDING（预计 2026-08-27 23:44 在 watgpu408 启动） |

三条使用标准 run ID `X4GD_full_s{0,1,2}_e25`；G 与 D 仍须等 GD 三条通过 §9 后提交。
seed 0 已写出 epoch 4 的 `ckpt/last.pt`（11,599,615 bytes）；相同 RUN_ID 的 requeue 会由
`resolve_resume()` 自动从 epoch 5 续跑，不会丢掉已完成的五个 epoch。

### 6. 全站空闲 GPU 定点探测与再次迁移（2026-08-27）

在三条 L40S 作业仍排队时，执行者按节点重新核对了所有 Slurm GRES，并对账面空闲卡提交了
短作业，而不是只依赖 `scontrol` 的分配数：

| 节点 | 账面空闲 | 定点探针 | 结论 |
|---|---:|---|---|
| watgpu108 | 多张 ADA6000 | `1522375` | 环境可 staging，但 CUDA 初始化失败，`cuda=False` |
| watgpu308 | 2 张 `schoolgpu` | `1522377` | RTX A6000，47.404GB，`cuda=True` |
| watgpu608 | 1 张未标型号 GPU | `1522378` | L40S，44.42GB，`cuda=True` |

因此取消仍为 PENDING 的 `1522230` / `1522231` / `1522232`，用相同标准 RUN_ID 重投；seed 0
保留原有 epoch 4 checkpoint：

| seed | 新 Job ID | 目标 / 内存 | 当前状态 |
|---:|---:|---|---|
| 0 | `1522383` | watgpu608 L40S / 48G | COMPLETED，机制门 PASS |
| 1 | `1522415` | watgpu308 L40S / 96G | 当时 PENDING；最终 MANIFEST 记录 L40S，启动时已到 epoch 25，只执行 no-op 门禁 |
| 2 | `1522381` | watgpu308 `schoolgpu`（型号无留档）/ 96G | RUNNING |

watgpu308 虽仍显示两张 GPU 未分配，但节点状态含 `PLANNED`，说明这些卡已被更高优先级作业规划；
因此当前只有 watgpu608 的一张卡真正立即交付给本实验。

watgpu608 的 batch shell、环境 staging 与 CUDA 均正常，但 `srun` job step 连续两次因节点通信
故障失败（`1522379` / `1522382`，ExitCode `105:0`），均未启动 Python 训练。脚本随后增加
`X4_DIRECT=1`：仍在同一 Slurm batch allocation 内运行完全相同的 Python 命令，只绕过故障的
job-step 通信层。替代作业 `1522383` 已成功读取原 `last.pt`，日志确认从 epoch 5 续跑。

### 7. GD seed 0 完成；seed 1/2 二次续跑（2026-08-27）

seed 0 的 `1522383` 在 watgpu608 完成，Slurm `COMPLETED 0:0`；训练墙钟 3150.6 秒、峰值
GPU 37.946GB，loss `22.7592 -> 15.1643`，无 NaN，`mechanism_gate.passed=true`。binding 的
图摘要、vocab 摘要与 family 数均通过 §9。

seed 1/2 曾在 watgpu308 获得 `gres/gpu:schoolgpu=1`，但 Slurm 实际 TimeLimit 只有 30 分钟；
两条分别以 `TIMEOUT` 结束，均已写出 epoch 9 的 `last.pt`，没有 MANIFEST，原始日志也没有 GPU
型号行，因此不能把同节点探针 `1522377` 抽到的 RTX A6000 型号外推给这两个训练 allocation。
它们随后以 90 分钟时限、
`X4_DIRECT=1` 重投到已验证的 watgpu608 L40S：

| seed | 当前 Job ID | resume 点 | 当前状态 / 预计启动（UTC） |
|---:|---:|---:|---|
| 1 | `1522429` | epoch 10 | PENDING；2026-08-28 02:58 |
| 2 | `1522430` | epoch 10 | PENDING；2026-08-28 04:29 |

此时 watgpu608 六张 GPU 已全部再次分配，因此两条等待的是该节点下一次释放窗口；checkpoint
使每条只需完成剩余 15 个 epoch。

seed 0 最终 25 epoch 完成：墙钟 3150.6 秒、峰值 GPU 37.946GB、loss
22.7592 → 15.1643、无 NaN，`mechanism_gate.passed=true`。seed 1 的首次 watgpu308 `schoolgpu` 作业
`1522380` 被 backfill 强制设为 30 分钟，在 epoch 9 checkpoint 后 TIMEOUT；seed 2
`1522381` 同样在 30 分钟 backfill 中运行。普通用户无权延长运行中的 TimeLimit，因此 seed 1
真正由 `last.pt` 从 epoch 10 续跑到 epoch 24 的作业是 watgpu608 上的 `1522429`；较早提交但
更晚启动的 `1522415` 读到的 checkpoint 已在 epoch 25，因此只执行 no-op 门禁并重写 MANIFEST。

截至 2026-08-27 20:06 UTC，`1522383` 已写出 epoch 9 checkpoint 并继续运行，尚无最终
MANIFEST。账户无权延长其运行中时限；若 90 分钟到时，按最近的 5-epoch checkpoint 再续跑。
为争取 watgpu308 的短 backfill，seed 1/2 的 PENDING 作业 `1522380` / `1522381` 已把时限
从 90 分钟下调为 30 分钟；训练目标仍是 25 epoch，超时后使用相同 RUN_ID 分段续跑。

### 8. GD 最新状态（2026-08-28 00:13 UTC）

- seed 0 已完成：`1522383`，25 epoch，wallclock 3150.6 秒，peak GPU 37.946GB，loss
  `22.7592 -> 15.1643`，机制门 PASS。
- seed 1 已写出完整 epoch 24 checkpoint、MANIFEST 和 PASS 机制门：累计 wallclock 2329.9 秒，
  peak GPU 37.964GB，loss `22.8314 -> 15.4913`。最后一层 Slurm allocation 记为 TIMEOUT，
  但训练产物在时限到达前已完整落盘。
- seed 2 的续跑 `1522430` 正在 watgpu608 运行，已从 epoch 10 恢复并写出 epoch 14 checkpoint；
  尚未生成最终 MANIFEST。误投的重复 PENDING 作业 `1522457` 已在启动前取消。

三条的图摘要均保持 `0e80b839...`，gamma 质量分别命中预注册值 0.5177 / 0.5184 / 0.5166。
G 与 D 继续等待 seed 2 的最终机制门，不提前提交。

## Codex的反馈

最终状态（2026-08-31）：**X4 的 GD / G / D 共 9 个正式 run 全部完成，9/9 的
`mechanism_gate.passed=true`，9/9 无 NaN，9/9 均包含 25 条 epoch loss。所需 checkpoint、
配置、指标、训练日志和执行证据已从 watGPU 下载并完成远端/本地 SHA-256 对验。**

### 结果与机制门

| run | 最终 Job ID | loss（epoch 0 -> 24） | manifest wallclock | peak GPU | 门禁 |
|---|---:|---:|---:|---:|---|
| `X4GD_full_s0_e25` | `1522383` | 22.7592 -> 15.1643 | 3150.6 s | 37.946 GB | PASS |
| `X4GD_full_s1_e25` | `1522415` | 22.8314 -> 15.4913 | 35.6 s* | 5.252 GB* | PASS |
| `X4GD_full_s2_e25` | `1522430` | 22.9725 -> 15.4189 | 2100.6 s | 37.927 GB | PASS |
| `X4G_full_s0_e25` | `1522702` | 22.7849 -> 16.4554 | 971.3 s | 37.949 GB | PASS |
| `X4G_full_s1_e25` | `1522703` | 22.1894 -> 16.6525 | 938.4 s | 37.968 GB | PASS |
| `X4G_full_s2_e25` | `1522704` | 22.4554 -> 16.5583 | 809.7 s | 37.936 GB | PASS |
| `X4D_full_s0_e25` | `1524613` | 22.2629 -> 14.7935 | 660.9 s** | 37.943 GB | PASS |
| `X4D_full_s1_e25` | `1524614` | 22.3808 -> 14.9880 | 803.0 s** | 37.968 GB | PASS |
| `X4D_full_s2_e25` | `1524617` | 22.6633 -> 15.1707 | 15.5 s* | 5.251 GB* | PASS |

`*` GD seed 1 和 D seed 2 的 25 epoch checkpoint 已在前一个 allocation 中完整落盘；随后
用于补写最终门禁的 no-op resume 覆盖了 MANIFEST 中的 `wallclock_s` 和 `peak_gpu_mem_gb`，因此
星号值只代表最后门禁段，不代表完整训练。GD seed 1 在覆盖前记录的完整值为 2329.9 s / 37.964 GB。
D seed 2 的训练进程曾在远端现场观察到 `nvidia-smi` 占用 43,030 MiB，但该命令输出当时没有
写入交付物，因此只能作为**未留档的现场观察**，不能由下述 `sacct.txt` 或 `slurm/` 独立复核；
交付报告也不把 no-op 的 5.251 GB 当作正式训练峰值。

`**` D seed 0/1 的 MANIFEST 墙钟是最后一个 resume 段；含 staging/排队内运行时间的三个
Slurm allocation 累计分别为 4354 s 和 4498 s。D seed 2 的相关 allocation 累计为 4491 s，
其中包括 watgpu408 上等待同节点 GPU 的时间和最后的 no-op 门禁段。

机制与配置复核结果：

- 每个 run 的 `n_models=3016439`、`epochs=25`、`num_families=41056`；图摘要均为
  `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`，vocab 摘要均为
  `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`。
- GD 配置含 `lake_gamma=0.5` 和 `global_n_datasets=128`；G 仅含 `lake_gamma=0.5`；D 仅含
  `global_n_datasets=128`。三组配置门全部符合 §9。
- `[lake logQ] mass_on_labeled`：GD/G 的 seed 0/1/2 为 0.5177 / 0.5184 / 0.5166；D 为
  0.0355 / 0.0369 / 0.0332，均命中预注册值。
- D 的三条 run 使用相同 RUN_ID 分段续跑：首段到 epoch 9，第二段到 epoch 19，末段到
  epoch 24。对应 Job 链为 seed 0 `1522762 -> 1524605 -> 1524613`，seed 1
  `1522765 -> 1524606 -> 1524614`，seed 2
  `1522766 -> 1524607 -> 1524615 -> 1524617`。其中 `1524615` 已写出 epoch 24 checkpoint，
  但在生成 MANIFEST 前被取消；`1524617` 从 epoch 25 no-op resume，成功生成最终 PASS 门禁。

### 硬件与环境异质性披露

X4 的训练 allocation 并非单一硬件。最终 MANIFEST 记录的 GPU 为：GD 三条 L40S；G 三条
RTX 6000 Ada Generation；D seed 0/2 为 RTX 6000 Ada，D seed 1 为 L40S。早期 GD seed 1/2
的 epoch 0--9 落在 watgpu308 的 `gres/gpu:schoolgpu=1`，但两段均 TIMEOUT，既无 MANIFEST，
训练日志也没有 GPU 型号行。watgpu308 探针 `1522377` 当次记录为 RTX A6000，而同节点同 GRES
的 no-op 作业 `1522415` 自身 MANIFEST 记录为 NVIDIA L40S、44.39GB，证明该 GRES 混有不同
型号；因此这两个早期训练 allocation 的具体型号记为**无留档**，不从探针推定。F8 中的 F6
三条基线则运行在 watgpu508/H200。

软件环境也有一项异质性：`X4D_full_s1_e25` 的最终 `resolved_config` 记录 Python 3.11.9，
其余八条为 Python 3.11.4；九条均为 PyTorch 2.12.0+cu130。同图、同代码、同配置和同 split 的
A 轴比较仍可进行，但写 `X4.md` 时必须同时披露 GPU 与 Python 差异，不应把它们描述为完全
同硬件/同环境复现。

### 产物路径

9 个远端 run 目录统一位于：

`/u801/x98liu/model_lake/runs/<RUN_ID>/`

9 个远端交付包统一位于：

`/u801/x98liu/x4_<RUN_ID>.tgz`

本地已按 §12 解压到：

`D:\research\model_lake\codes\ModelLakeFishing\docs\1M\X4_runs\<RUN_ID>\`

其中每个正式 checkpoint 的成功路径为：

`D:\research\model_lake\codes\ModelLakeFishing\docs\1M\X4_runs\<RUN_ID>\ckpt\last.pt`

`<RUN_ID>` 的完整取值为：

- `X4GD_full_s0_e25`、`X4GD_full_s1_e25`、`X4GD_full_s2_e25`
- `X4G_full_s0_e25`、`X4G_full_s1_e25`、`X4G_full_s2_e25`
- `X4D_full_s0_e25`、`X4D_full_s1_e25`、`X4D_full_s2_e25`

原始下载包保存在：

`D:\research\model_lake\codes\ModelLakeFishing\docs\1M\X4_runs\_packages\`

执行证据保存在：

- `docs/1M/X4_runs/x4_execution/sacct.txt`：所有正式训练 allocation 的 Slurm 状态、节点、
  起止时间、Elapsed、MaxRSS、AllocTRES 与 ExitCode。
- `docs/1M/X4_runs/x4_execution/slurm/`：训练、冒烟和环境/GPU 探针的原始 Slurm 日志。
- `docs/1M/X4_runs/x4_execution/code_sha256.txt`：远端执行代码摘要。
- `docs/1M/X4_runs/x4_execution/package_sha256.txt`：9 个远端交付包摘要。
- `docs/1M/X4_runs/_packages/x4_execution.tgz`：上述执行证据的原始下载包；本地 SHA-256 为
  `db90ff391deec68b0028921a4cffc87840be2484021b1cf998121e36a359a5a9`。

### 下载包 SHA-256

| 包 | SHA-256 |
|---|---|
| `x4_X4GD_full_s0_e25.tgz` | `cc7983ed4c6ec8de443a8e4350757202cd94c1aaf918aafe8c11b08c403384be` |
| `x4_X4GD_full_s1_e25.tgz` | `0dee63fb925d759fc3dc6585a5ab34d355975572afa63e69f49919ef22edef98` |
| `x4_X4GD_full_s2_e25.tgz` | `d4ea6141a994b9e6e274b61d52441c877daec549f8b55a52b976556d8d118ce1` |
| `x4_X4G_full_s0_e25.tgz` | `d329efec287efb3a8bf7a6fbc6fe19f59415ef1a29587dcbcafb0bd5780afe1e` |
| `x4_X4G_full_s1_e25.tgz` | `0033b395e5ec5d47ded045a55e5acad040ec2adb53b353b0a5fafa95dedb2a90` |
| `x4_X4G_full_s2_e25.tgz` | `149be46b0b3f3c937440d4c16b613aa4568237008b26da07f76991e06d8d2584` |
| `x4_X4D_full_s0_e25.tgz` | `95dfe67fde9d96fad572df85a7ac3e84fb6d90a6226f9097dc62f0d71318e8f3` |
| `x4_X4D_full_s1_e25.tgz` | `bdda893969072db2b3c54cea0da7f2fed1e16446aca7455faffaa750586a714e` |
| `x4_X4D_full_s2_e25.tgz` | `1dda086d7eea80a8d5841eb6203b0ed4d603d56cfc1c266939c177fa1c2e7a71` |

本地重新计算的 9 个摘要与远端 `package_sha256.txt` 逐项一致。每个解压目录也再次验证了
`MANIFEST.json`、`metrics/train_history.json`、`metadata/resolved_config.json`、
`stdout/train.log`、`ckpt/last.pt` 和 `ckpt/family_vocab.csv` 六类必需产物全部存在。

本节完成 X4GPU §11 的交付和 §12 要求的解包。

§12 的本地部分已于 2026-09-01 执行完毕：9 个 run 的 `embed` 与 `metrics` 导出写在
`data1m/exports_x4/`，三条臂的 A 轴与 C 轴报告写在 `data1m/metrics_x4/{GD,G,D}/`，
副本在 `docs/1M/X4_runs/reports/`。结论、与 §2 预注册的逐条对照、以及未做的部分
（`index` / `curve` 导出，B、D、P、S 四轴）写在 [`X4.md`](X4.md)。
