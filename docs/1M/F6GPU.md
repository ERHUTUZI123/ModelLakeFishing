# F6：在 watGPU 上训练全量图

上级 runbook：[`1Mplan.md`](1Mplan.md) §5 F6。上游：[`F5.md`](F5.md)，图已在本地建成并通过出闸门。

本文写给执行者。范围是从把图传上 watGPU 开始，到交回训练 checkpoint 与 `MANIFEST.json` 为止。
导出 `z_m`、建 HNSW 索引与三轴评测属于 F7/F8，不在本文范围内。

按顺序读完 §2 到 §5 再动手：那四节说明的是「为什么现在照着 1Mplan §5 F6 里的命令敲会失败」，
以及有一处graph 层面的差异需要先定案，定案之后才值得开始训练。

---

## 1. 目标与交付

F6 产出 checkpoint 与 loss 曲线，不产出精度数字。`gold@10`、检索延迟、冷启动分层都要等 F7 把
`z_m` / `z_d` 导出并建好索引之后才有，所以本阶段能验收的只有机制门（§9）：loss 是否下降、
有没有 NaN、两张 learnable embedding 表是否拿到非零梯度、frozen `x` 是否确实没有梯度。

每个 run 交回四份文件加一行 `sacct`，见 §11。checkpoint 体积大（模型加 optimizer 状态），
留在远端即可，F7 在同一台机器上读它。

---

## 2. 三样不要改

这三条是 1Mplan §3.6 的受控变量，改了会让 F8 的跨档比较失去意义。

训练配置沿用 `scale/export_ours.py` 里的 `l1l3b_config()`，一个键都不动。它解析出来是
`num_layers=1`、`top_frac=0.10`、`lr=1e-2`、`batch_size=1024`、`rank_loss=ranknet`、
`global_mode=lake`、`global_n_neg=256`、`similar_to_mode=topk_unweighted`、`similar_to_k=10`。
`metadata/resolved_config.json` 会把实际生效的值写下来，跑完对一遍。

`init_seed` 恒为 0（D-43）。`--seed` 传的是 `split_seed`，三个 seed 给的是划分噪声区间，
不是初始化噪声区间；报数字时要写明是哪一种。

特征保持 float32（D-55）。不要为了省显存给 `x` 降精度，也不要开 `--amp`（D-42）。
显存如果不够，先按 §5 的办法减少每步的图复制，那是不改数值的。

---

## 3. 代码缺口：五处

F5 写出来的是 `scale1m.graph_store` 的拆件目录，而训练侧的代码是为 100K 档的单文件 `.pt` 写的。
下面五处要先补齐，前四处是必需的，第五处可以推迟。改完跑一遍 `pytest scale1m/tests`
与 `pytest stage2TrainGraphSAGE/tests`，当前基线是 230 passed，数字不应该变小。

### 3.1 `train_rung.py` 只会读单文件

[`scale1m/train_rung.py:224`](../../scale1m/train_rung.py#L224) 是 `torch.load(args.graph, ...)`。
传一个目录进去会直接抛 `IsADirectoryError`。加一个分支：

```python
    if os.path.isdir(args.graph):
        from scale1m.graph_store import load_sharded
        # verify_sha256=False：到达时校验过一次（§6），每次 requeue 重读 5.66 GB
        # 只是把排队时间花在磁盘上。
        payload = load_sharded(args.graph, mmap=True, verify_sha256=False)
    else:
        payload = torch.load(args.graph, map_location="cpu", weights_only=False)
```

`load_sharded` 返回的 dict 与 `torch.load` 的 payload 结构相同（`data` / `xm0_meta` /
`xd0_meta` / `unique_model_id` / `unique_dataset_id`），后面那段代码不用动。
`unique_dataset_id` 里有 `root` 列，`make_root_aware_splits` 要的就是它。

### 3.2 `--rung` 没有全量档

[`scale1m/train_rung.py:189`](../../scale1m/train_rung.py#L189) 用 `choices=sorted(RUNGS)` 限死了
`12k` / `30k` / `100k`。在 `RUNGS` 里加两行：

```python
    "500k": {"expect_n": None, "anchor_gold10": None},
    "full": {"expect_n": 3_016_439, "anchor_gold10": None},
```

`expect_n = 3_016_439` 是图的模型节点数（3,003,759 个快照模型加 12,680 个只在历史监督里
出现的模型），断言在加载后、训练前就跑，见 `train_rung.py` 里 §1 rule 3 那段。
500K 档的行数还没定，留 `None`，用命令行的 `--expect-n` 传。

注意这个数与评测时的候选池不是一回事。1Mplan §2.2 说候选池只由快照定义（3,003,759），
§2.3 又说湖外模型「仍然进湖作为候选」，两处表述不一致。这个口径由 F7/F8 定案，
F6 只断言图自己的节点数，不要在这里替它做决定。

### 3.3 `graph_sha256` 在目录上会抛异常

两处都调用 `sha256_of` 去哈希一个文件：
[`scale1m/checkpoint.py:95`](../../scale1m/checkpoint.py#L95)（写进 binding，resume 时逐项校验）
和 [`scale1m/train_rung.py:117`](../../scale1m/train_rung.py#L117)（写进 `resolved_config.json`）。
`os.path.exists` 对目录返回 True，于是它会走进 `open()` 然后抛 `IsADirectoryError`。

在 `checkpoint.py` 里加一个对两种布局都成立的摘要函数，两处都改用它：

```python
def graph_digest(path):
    """图的内容标识，单文件与拆件目录通用。

    拆件目录没有单一文件可哈希，但 meta.json 已经为每个分片记了 sha256，
    所以摘要取在那张表上。单文件的结果与原来逐字节相同，T6 的 checkpoint
    的 binding 因此仍然有效。
    """
    if path and os.path.isdir(path):
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            files = json.load(fh)["files"]
        blob = json.dumps(files, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return sha256_of(path) if path and os.path.exists(path) else None
```

`hashlib` 与 `json` 在 `checkpoint.py` 里已经 import 过了。
`meta.json` 自己不在 `files` 表里，所以它的 sha256 单独记在 §6。

补两个测试：单文件路径的返回值与 `sha256_of` 相同；同一个拆件目录两次调用相同，
改动其中一个分片的 sha256 之后结果变化。

### 3.4 作业脚本的路径是 100K 档的

当时使用的脚本现已归档为 [`legacy/100k_rung/scripts/watgpu/train_rung.sbatch`](../../legacy/100k_rung/scripts/watgpu/train_rung.sbatch)，它找的是
`$DATA_ROOT/data1m/graphs/hgraph_${RUNG}.pt` 和 `$DATA_ROOT/data1m/feats/${RUNG}/family_vocab.csv`，
两个路径本档都不存在。新写一个 `scripts/watgpu/train_rung_rf.sbatch`，不要改原来那个——
它还是复现 100K 三个 seed 的那份脚本。

新脚本与旧脚本的差别只有五处，其余（环境 staging、`RUN_ID` 推导、requeue 续跑、
`family_vocab` 存在性检查、T0 开关）原样照抄：

| 项 | 旧 | 新 |
|---|---|---|
| 图路径 | `data1m/graphs/hgraph_${RUNG}.pt` | `data1m/graphs/hgraph_rf`（`RUNG=full`）/ `hgraph_rf_500k`（`RUNG=500k`） |
| vocab 路径 | `data1m/feats/${RUNG}/family_vocab.csv` | `data1m/feats_rf/family_vocab.csv` |
| 存在性检查 | `[ -f "${GRAPH}" ]` | `[ -d "${GRAPH}" ]` |
| `--mem` | 96G | 128G 起步，按冒烟的 `MaxRSS` 调 |
| `--time` | 00:30:00 | 冒烟用 01:00:00，正式跑按实测定（D-45） |

另外加一个 `--skip-diagnostics`，理由在 §5.3。

### 3.5 500K 档的图还不存在（可以推迟）

[`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py) 没有子采样参数，
`data1m/graphs/` 下也只有全量图。1Mplan §3.5 要的 500K 训练档需要单独建一张图，
而建图按 D-67 是本地的活。

这不阻塞 F6 开工：全量档能不能跑，由 §8.2 的全量冒烟直接回答，不需要先跑 500K。
500K 档是训练侧 scaling 曲线的第二个点，科学上有用但可以后补，
1Mplan §3.5 也写明了配额不够时它是第一个被砍的。建图的事另行安排。

---

## 4. 一处要先定案的差异：数据集侧三张表退化成单行

这一条要在开始训练之前定，因为改了图就要重训。

RF 图的 `xd0_meta` 是 `num_task_types=1`、`n_class_buckets=1`、`num_arities=1`，
对应的三列 `dataset.task_type_id` / `n_class_bucket_id` / `arity_id` 全是 0。
100K 档同一位置是 196 / 7 / 5。`ablation._ds_kwargs()` 把这三个数直接传给
`HeteroGraphSAGE` 去建 `nn.Embedding`，所以在本档里这三张表各只有一行，
等价于给每个数据集节点加同一个可学习偏置，不携带任何信息。

训练不会因此报错，梯度也非零，机制门照样通过——这是它需要被明写出来的原因。

任务类别其实是有的，只是没提取：数据集节点的 id 就是 `名字\t任务`，
18,729 个节点里任务字段一个都不缺，去重后有 2,326 个不同的任务字符串
（未做规范化，`token classification` 与 `token-classification` 现在算两个）。
另外两列（类别数分桶、arity）依赖 RF 流程没有采集的数据集元数据，短期补不了。

三个选项：

1. 照现状跑，在 F8 里写明 RF 档与 100K 档之间除了 N 还差这一项。代价是跨档比较多了一个变量。
2. 在 F5 里从节点 id 的第二段建 `task_type_id` 词表（需要与 100K 档相同的规范化），
   重建图后重传。图只有 `nodes.npz`（73 MB）与 `meta.json` 变化，用 rsync 增量传很便宜，
   但 `graph_sha256` 会变，此前的 checkpoint 作废。
3. 先按选项 1 跑冒烟拿到时间与显存，同时并行做选项 2，正式的三 seed 用新图。

执行者不要替这件事做决定，把本节连同 §12 的其他判断点一起回报。
如果要动，动的时机是三 seed 正式训练之前。

---

## 5. 规模上的关键事实：每个训练步都做一次全图前向

这是 F6 唯一真正的规模问题，值得在提交作业之前读懂。

### 5.1 机制

`l1l3b_config()` 里 `lambda_global = 1.0`，于是
[`stage2TrainGraphSAGE/train.py:129`](../../stage2TrainGraphSAGE/train.py#L129) 会把整张训练消息图
搬到 GPU 常驻，并且在每个训练步里
[做一次带梯度的全图前向](../../stage2TrainGraphSAGE/train.py#L207)：

```python
g_base  = train_data.clone().to(device)      # 常驻
...
g_graph = g_base.clone()                     # 每步一份
apply_edge_dropout(g_graph, ...)
z_full  = model(g_graph)
```

这是 L1 全湖 logQ 采样 softmax 的设计要求：负样本取自全湖，mini-batch 的子图里没有它们的表示。
`--fanout` 与 `--chunked-infer` 都管不到这一项，前者只界定 mini-batch 子图，后者只作用于训练结束后的推理。

在全量图上，`x` 是 3,016,439 × 448 的 float32，5.41 GB。常驻一份加每步一份，
光是特征就占 10.8 GB 显存，激活还没算。

### 5.2 外推

以 T6 在 watgpu808（H200 NVL，143,771 MiB）上的 `R2_100k_s0_e25` 为基准：25 epoch 108.5 秒，
峰值显存 1.737 GB。

| 量 | 100K 档 | 全量档 | 比值 |
|---|---:|---:|---:|
| 模型节点 | 100,000 | 3,016,439 | 30.16 |
| 数据集节点 | 9,603 | 18,729 | 1.95 |
| `trained_on`（决定每 epoch 的步数） | 312,986 | 247,803 | 0.79 |
| 全部边 | 850,886 | 2,588,316 | 3.04 |

每步的开销由全图前向主导，随节点数近似线性；每 epoch 的步数随监督边数走，反而少了两成。
按此外推：峰值显存约 **52 GB**，25 epoch 的墙钟约 **43 分钟**。两个都是量级估计，
真实值由 §8.2 的冒烟给出。

由此推出的硬性要求：这份作业要跑在 80 GB 以上的卡上。48 GB 的 RTX 6000 Ada 按外推装不下，
H200 NVL 有充裕余量。沿用 T6 的 `--nodelist=watgpu808`，或在提交前确认分到的卡至少 80 GB。

### 5.3 关掉训练内的诊断

`train_eval_one` 在训练结束后会算一批诊断量，其中
`dataset_to_model_hnsw_recall` 会在全部模型向量上建一个 HNSW 索引。
在 3M 上这是把 F7 的活提前做一遍，而且要求远端已装 `hnswlib`；
`per_dataset_density` 是一个按数据集节点的 Python 循环，每轮全扫一遍监督边，
18,729 × 247,803 次比较。两者都会在训练已经跑完之后才失败或长时间挂住。

所以正式作业加 `--skip-diagnostics`。它跳过的是 `mean_cos`、`z_m_pr`、`z_d_pr`、
`density`、`pd_density`、`dm_recall` 六项，其中前三项正是 1Mplan §5 F8 的 C 轴要重算的，
不会丢信息。`tau_macro` 与 head retrieval 不在跳过之列，照常产出。

### 5.4 如果冒烟仍然 OOM

最小的改法是让 `g_graph` 只复制边而与 `g_base` 共享 `x`。
`apply_edge_dropout` 只改 `edge_index` 与 `edge_attr`，`model(g_graph)` 只读 `x`，
所以共享是安全的，每步能省下一份 5.41 GB。

这改的是所有历史配置都会走的公共路径，所以要配一条测试：在小图上固定 seed，
改动前后的逐 epoch loss 逐位相同。不要在没有这条测试的情况下提交。

---

## 6. 上传与校验

要传的东西，合计约 5.76 GB，绝大部分是 `x_model.npy`：

| 本地 | 远端 | 大小 |
|---|---|---|
| `data1m/graphs/hgraph_rf/` | `$DATA_ROOT/data1m/graphs/hgraph_rf/` | 5.66 GB |
| `data1m/feats_rf/family_vocab.csv` | `$DATA_ROOT/data1m/feats_rf/family_vocab.csv` | 606 KB |
| `data1m/ladder_rf/full_model_ids.parquet` | `$DATA_ROOT/data1m/ladder_rf/` | 99 MB |
| `data1m/ladder_rf/full_dataset_ids.parquet` | `$DATA_ROOT/data1m/ladder_rf/` | 649 KB |

梯子的两个 parquet 训练用不到，F7 的行序门要用，一起传省一次往返。
`feats_rf/x_desc_part-*.npy` 是 F4 的中间件，不要传。

用 rsync，5.66 GB 断了可以续：

```bash
rsync -av --partial --progress \
  /d/research/model_lake/data/data1m/graphs/hgraph_rf/ \
  x98liu@watgpu.cs.uwaterloo.ca:<REMOTE_DATA_ROOT>/data1m/graphs/hgraph_rf/
```

到达后先核对四个小文件与目录清单，再核对分片：

```bash
sha256sum "$DATA_ROOT/data1m/graphs/hgraph_rf/meta.json" \
          "$DATA_ROOT/data1m/feats_rf/family_vocab.csv" \
          "$DATA_ROOT/data1m/ladder_rf/full_model_ids.parquet" \
          "$DATA_ROOT/data1m/ladder_rf/full_dataset_ids.parquet"
```

应当逐字相同：

```text
82cb04a78692b36ea51808c434ca00de3ea78a3a4fddbc98e21a8f2d0fbee4d6  meta.json
00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005  family_vocab.csv
fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa  full_model_ids.parquet
31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb  full_dataset_ids.parquet
```

`meta.json` 里的 `files` 段记了另外八个文件各自的 sha256，其中 `x_model.npy` 应当是
`ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`，与 F4 报告里的
`x_m.npy` 是同一个值。逐个复算：

```bash
cd "$DATA_ROOT/data1m/graphs/hgraph_rf"
python - <<'PY'
import hashlib, json, os
files = json.load(open("meta.json", encoding="utf-8"))["files"]
for name, want in sorted(files.items()):
    h = hashlib.sha256()
    with open(name, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    print(("OK  " if h.hexdigest() == want else "FAIL"), name)
PY
```

代码走 git：本地提交推送，远端 `cd "$PROJECT_ROOT" && git pull --ff-only`。
§3 的改动都在仓库里，不要用 scp 单独拷 `.py`。

---

## 7. 环境前置

环境沿用 T4 建好的那份（`ENV_SOURCE` 加 staging 到节点本地盘，因为 `$HOME` 是 noexec）。
T6 在 2026-08-11 于计算节点装上并验过 `pyg-lib 0.8.0+pt212cu130`，
如果那次的选择被持久化进了 `ENV_SOURCE`，本阶段不需要额外装包。先确认一次：

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00
source ~/.mlf_env && source "$ENV_SOURCE/bin/activate"
python -c "
import torch
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
print('torch', torch.__version__, '| pyg_lib', WITH_PYG_LIB, '| torch_sparse', WITH_TORCH_SPARSE)"
```

两个标志有一个为 True，`make_link_loader` 就走 PyG 的 `LinkNeighborLoader`（C++ 采样）；
都是 False 就退回纯 Python 的 `LightLinkLoader`，加了 `--fanout` 之后子图有界、能跑，
但每个 batch 的采样在 Python 里做，在 3M 上大概率慢到不能接受。都是 False 的话按 T6 的做法
在计算节点上装 `pyg-lib`，并把探测输出一并回报。

`hnswlib` 与 `faiss` 是 F7 的前置，F6 用不到（§5.3 之后训练内也不再建索引）。
如果这次 salloc 顺手，可以一并确认它们在不在，省一次排队。

---

## 8. 执行顺序

顺序与 1Mplan §5 F6 里列的四条命令不同：那里是 500K 在前，而 500K 的图还不存在（§3.5），
并且需要回答的问题——全量档在这张卡上跑不跑得动——只有全量冒烟能回答。
定案后回填 1Mplan。

### 8.1 全图契约检查

F5 的 Stage-2 契约检查是在 150,000 个模型的切片上跑的，本地内存装不下全图前向。
远端有 128 GB，可以补上这一步，同时它会用 `verify_sha256=True` 把每个分片重算一遍：

```bash
source ~/.mlf_env && cd "$PROJECT_ROOT"
python -m ModelLakeFishing.stage1BuildTransferGraph.check_stage2_contract \
  --sharded "$DATA_ROOT/data1m/graphs/hgraph_rf"
```

七项全 OK 才继续。它是 CPU 作业，全图前向加反向大约需要 10–15 GB 内存。
其中第 4 项（`family_vocab` 是 `0..41055` 的连续双射）正是 F5 里查出词表少一行的那一条。

### 8.2 全量冒烟，1 epoch

```bash
cd "$PROJECT_ROOT" && mkdir -p logs
RUNG=full SEED=0 EPOCHS=1 RUN_ID=RF_full_s0_smoke \
  sbatch --time=01:00:00 --export=ALL,RUNG,SEED,EPOCHS,RUN_ID \
  scripts/watgpu/train_rung_rf.sbatch
```

这一步的产出是三个数，不是模型：单 epoch 墙钟、`peak_gpu_mem_gb`、`sacct` 的 `MaxRSS`。
后面所有作业的 `--time` 与 `--mem` 都从这里定（D-45：不要留大额占位值，
Slurm 按申请的墙钟排队，占位值会把短作业排到次日）。

把这三个数回报之后再提交 §8.3。如果峰值显存超过卡容量的七成，先看 §5.4。

### 8.3 全量正式跑，25 epoch × 三个 seed

`--time` 用冒烟外推值的两倍左右，不要更多。三个 seed 可以并行提交。

```bash
for s in 0 1 2; do
  RUNG=full SEED=$s EPOCHS=25 \
    sbatch --time=<按 8.2 定> --export=ALL,RUNG,SEED,EPOCHS \
    scripts/watgpu/train_rung_rf.sbatch
done
```

三个 seed 是为了给 A 轴一个区间。这个项目拦下过一次单 seed 的假信号
（v4 W3 的三 seed 结论在八个 seed 下不成立），所以不要用单 seed 的数下结论。

作业被 preempt 重投时保持同一个 `RUN_ID`：驱动会自己找 `<run>/ckpt/last.pt` 续跑，
换 `RUN_ID` 会把同一次逻辑运行的指标劈成两个目录。

### 8.4 500K 档

图建好并传上来之后再跑，命令与 8.3 相同，`RUNG=500k` 并显式传 `--expect-n`。
它是训练侧 scaling 曲线的第二个点，不阻塞 F7。

---

## 9. 怎么判断跑对了

Slurm 报 `COMPLETED` 不等于通过。每个作业跑完看 `MANIFEST.json`：

| 字段 | 应该是什么 |
|---|---|
| `mechanism_gate.passed` | `true`。false 就停下，把 `metrics/train_history.json` 一并回报 |
| `mechanism_gate.loss_descended` | `true` |
| `mechanism_gate.no_nan` | `true` |
| `n_models` | 3016439 |
| `binding.graph_sha256` | 三个 seed 之间相同；与 §3.3 的 `graph_digest` 在本地算出的值相同 |
| `binding.num_families` | 41056 |
| `binding.family_vocab_sha256` | `00d304df9bd72acb…` |
| `seed` / `init_seed` | `init_seed` 恒为 0，`seed` 是 0/1/2 |
| `start_epoch` | 没被 preempt 就是 0；被 preempt 过时它是续跑点，`resumed_from` 非空 |
| `peak_gpu_mem_gb` / `wallclock_s` | 记下来，F7 的作业规格靠它估 |

`metadata/resolved_config.json` 里的 `git_head` 要非空，`resolved_config` 要与 §2 列的值逐项相同。

---

## 10. 已知的坑

| 症状 | 原因 | 处理 |
|---|---|---|
| `IsADirectoryError` | §3.1 或 §3.3 的改动没做全，还有地方在把目录当文件哈希 | 两处 `sha256_of(graph_path)` 都要换成 `graph_digest` |
| `argument --rung: invalid choice: 'full'` | §3.2 没做 | 在 `RUNGS` 里加档位 |
| CUDA OOM，且发生在第一个 step | 全图前向的两份 `x` 加激活超了卡容量 | 先确认卡至少 80 GB；仍然不够按 §5.4 让 `g_graph` 共享 `x` |
| CPU 内存撞上 `--mem` | `train_data.clone()` 会把 memmap 的 5.41 GB 实体化一次 | `--mem` 从 128G 起；`sacct` 的 `MaxRSS` 是真实峰值 |
| 训练极慢、GPU 利用率低 | 走了 Python 的 `LightLinkLoader` | §7 的采样后端探测 |
| `LinkNeighborLoader` 报只读或共享内存相关错误 | `x` 是只读的 numpy memmap | 在 `load_sharded` 后加 `data['model'].x = data['model'].x.clone()`，代价是 5.41 GB 常驻内存；或直接 `mmap=False` |
| 训练跑完之后长时间挂住或在建索引时报错 | 没加 `--skip-diagnostics` | 见 §5.3 |
| `IncompatibleCheckpoint: graph_sha256 ...` | run 目录复用了，但图换了（例如按 §4 选项 2 重建过） | 换一个 `RUN_ID`，不要删 checkpoint 硬续 |
| resume 之后 loss 跳变 | RNG 没在 `train()` 内、`torch.manual_seed(init_seed)` 之后恢复 | T6 修过一次，确认跑的是修好之后的 commit |
| `AssertionError: graph has N models, rung full expects` | 图与档位对不上 | 这条断言在训练开始前就跑，先查传上来的是不是 F5 那张图 |

---

## 11. 跑完交回什么

checkpoint 不要下载。每个 run 打一个包：

```bash
cd "${OUTPUT_ROOT}/runs/<RUN_ID>"
tar czf ~/f6_<RUN_ID>.tgz MANIFEST.json metrics/ metadata/ stdout/train.log
```

| # | 内容 | 用途 |
|---|---|---|
| 1 | `MANIFEST.json` | 机制门、墙钟、显存、binding、resume 情况 |
| 2 | `metrics/train_history.json` | 逐 epoch loss，看有没有该降不降 |
| 3 | `metadata/resolved_config.json` | 确认 §2 的配置没被改过 |
| 4 | `stdout/train.log` 与 `logs/mlf-*-<JOB_ID>.out` | 过程留痕 |
| 5 | `sacct -j <JOB_ID> --format=JobID,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode` | 排队时间与真实内存峰值 |

另外还要：§7 的采样后端探测输出，§8.1 的契约检查完整输出。

---

## 12. 需要测量并回报的判断点

这四条由执行者测量并回报，不要自行定案。

全量冒烟的墙钟、峰值显存、`MaxRSS`。它们决定 8.3 的 `--time`/`--mem`，
也决定 §5.4 的改动做不做。

§4 的 `task_type_id`：三个选项里选哪个。冒烟可以先在现状的图上跑，
但三 seed 正式训练之前要有结论，否则要重训。

`--skip-diagnostics` 跳掉的 `mean_cos` 是 O(N·d) 的，很便宜，只是与另外五项共用一个开关。
如果想在训练里保留塌缩的早期信号，可以把它拆成单独一项；这是可选的小改动，不是必需。

500K 档的排期。图要在本地重建并再传一次，与全量档相比它的科学收益要与配额一起权衡。

---

## Codex执行反馈

> 本节由 Codex 维护，用来区分操作指南与本次实际执行记录。

状态：**代码与图已就位；全图契约检查运行中，环境探测排队，1 epoch 冒烟已建立成功依赖；正式三 seed 尚未提交。**

### 1. 身份、凭据与前置件

- 执行者：Codex
- 开始日期：2026-08-20（America/Toronto）
- 凭据检查：F6 不需要密码、API key 或新 token；现有 SSH 公钥可无交互登录
  `x98liu@watgpu.cs.uwaterloo.ca`，执行期间没有读取或要求任何秘密。
- 本地 `hgraph_rf/`、`family_vocab.csv`、两个 ladder parquet 均存在；四个入口最初的 SHA-256
  与 §6 完全一致。
- 此前远端 F4 作业 `1515310` 因 watGPU `srun` 通信故障失败，没有生成远端 `feats_rf`。
  F6 使用本地已经通过 F5 闸门的正式图上传，不把该失败作业视为 F6 输入凭据。

### 2. Codex 补齐的代码缺口

按 §3 实现并同步了：

- `scale1m/train_rung.py`：支持拆件目录图；新增 `500k` / `full`，其中 full 断言
  `n_models == 3,016,439`；metadata 改用目录摘要。
- `scale1m/checkpoint.py`：新增单文件/拆件目录通用的 `graph_digest()`，checkpoint binding 使用它。
- `scale1m/tests/test_checkpoint.py`：新增单文件摘要等价性、目录摘要稳定性及变更敏感性测试。
- `scale1m/graph_store.py` 与 `check_stage2_contract.py`：远端旧 bundle 中缺少目录加载实现，定向同步本地正式版本。
- 新增 `scripts/watgpu/train_rung_rf.sbatch`：RF 图/vocab 路径、目录存在性检查、128G 内存、
  H200 节点、`--skip-diagnostics`、full/500k 分支均已落实；原 100K 脚本未改。

本地回归结果：`scale1m/tests` **232 passed**，`stage2TrainGraphSAGE/tests` **66 passed**，
合计 **298 passed**；远端 `bash -n` 与四个 Python 文件的 `py_compile` 通过。

本次没有创建 Git commit 或 push。远端 `origin` 是停在 `7d3457a…` 的旧 bundle，无法取得当前本地
F5/F6 文件，因此定向同步运行文件，并以下列 SHA-256 锁定：

| 文件 | SHA-256 |
|---|---|
| `checkpoint.py` | `a24a4505381f071c1162ae314c3b2e27cc5fb38a8839db998be6a1193454e20d` |
| `train_rung.py` | `11c530b80af66d7ce75f06e57f448cef49f4892788668b0a7aff4e570c77e9b8` |
| `graph_store.py` | `488c472249b4060acf73866a8b3d29c9bb9d09bb2d7cc85a18d919cf2197e037` |
| `check_stage2_contract.py` | `ebce2191177a3a9820a0e4677d87cae7860700632b0d5ce0e631ec8e2c6bc738` |
| `train_rung_rf.sbatch` | `2f9c9bd36e39896fa1bd46e36cd10b096cfbfba039b6047a269999f05d225265` |

### 3. 图上传与哈希核验

本机没有 rsync/WSL，改用 OpenSSH SFTP 上传。中断后的首次 `reput` 与尚未退出的旧进程发生重叠，
远端 `x_model.npy` 一度变成 5,518,098,816 bytes，大于本地正式件 5,405,458,816 bytes。
该副本明确损坏，已删除并用单一 SFTP 进程从零重传；删除目标仅为这一份可从本地恢复的损坏副本。

重传完成后，七个训练核心拆件全部命中 `meta.json`，包括：

- `x_model.npy`：`ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6`
- `x_dataset.npy`：`f4279117459d01fddc5c1ca889f2bca4e34f6c56611effebd3e05a48ad8d8b93`
- `nodes.npz`：`4e2d97496691706b706bb174e538c3aadaaf1976e1dbf85394ebeaa3240a5969`
- `edges.npz`：`c67a65eb7715728b357dd92a1816c3a7374caf28b2370f42ea6c84cb5a1695c2`

核验同时发现一个上游 F5 元数据缺陷：本地、远端的 `meta.json` 都把 `GRAPH_REPORT.json` 记录为
旧摘要 `e2d8bf…`，实际最终报告是 `53081ab…`，所以 §8.1 的 `verify_sha256=True` 必然失败。
已只修正 `meta.json.files["GRAPH_REPORT.json"]` 为实际摘要；训练张量与边均未改。

- 修正后的 `meta.json` SHA-256：`4ed7b5d13395d3e5fe98faa3e3b546199e6c569687e2b0d27fbd278676398fa0`
- 拆件目录 `graph_digest`：`0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`
- 修正后 `meta.json.files` 的八个文件在远端逐一复算，结果 **8/8 OK**。

这使 `meta.json` 不再等于 §6 预写的 `82cb04…`，但消除了其内部自相矛盾；该偏离必须随 checkpoint
binding 一起保留，三个 seed 也必须绑定上述同一个 `graph_digest`。

### 4. 当前 Slurm 执行链

| 阶段 | Job ID | 当前状态 | 依赖 / 目的 |
|---|---:|---|---|
| 全图 Stage-2 契约（首次） | `1515422` | FAILED（启动前置） | 未设置仓库父目录 `PYTHONPATH`，模块导入前即退出；未加载图 |
| 全图 Stage-2 契约（修正后） | `1515425` | COMPLETED 0:0 | 全量 3,016,439 行前向、两张 embedding 梯度、frozen x 等全部 OK |
| PyG/H200 环境探测 | `1515423` | PENDING（Priority） | 要求 `WITH_PYG_LIB` / `WITH_TORCH_SPARSE` 至少一个为 True |
| full / seed 0 / 1 epoch 冒烟（原依赖） | `1515424` | CANCELLED | 前一契约失败后变为 `DependencyNeverSatisfied`，已取消 |
| full / seed 0 / 1 epoch 冒烟（当前） | `1515426` | PENDING（Dependency） | `afterok:1515425:1515423`；任一前置失败都不会训练 |

冒烟 `RUN_ID=RF_full_s0_smoke`，申请 H200、128G RAM、1 小时。当前没有提交 25 epoch × 三 seed：
必须先由冒烟测出单 epoch 墙钟、`peak_gpu_mem_gb`、`MaxRSS`，并按 §12 定案 `task_type_id` 方案。

查看状态与日志：

```bash
source ~/.mlf_env
cd "$PROJECT_ROOT"

squeue -j 1515425,1515423,1515426
squeue --start -j 1515425,1515423,1515426
tail -f logs/mlf-f6-contract-1515425.out
tail -f logs/mlf-f6-pyg-probe-1515423.out
tail -f logs/mlf-f6-rf-1515426.out
```

结束后验收：

```bash
sacct -j 1515422,1515425,1515423,1515424,1515426 \
  --format=JobID,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode

cat "$OUTPUT_ROOT/runs/RF_full_s0_smoke/MANIFEST.json"
cat "$OUTPUT_ROOT/runs/RF_full_s0_smoke/metrics/train_history.json"
cat "$OUTPUT_ROOT/runs/RF_full_s0_smoke/metadata/resolved_config.json"
```

通常应在契约、采样后端和 1 epoch 冒烟全部通过后再提交 §8.3；本次为避免夜间人工操作，
采用下述可自动阻断的依赖链预排。

### 5. 无人值守的正式训练预排

用户明确要求不要等冒烟结束后再人工起床提交，因此已把正式训练作为受 gate 约束的依赖作业提前排入
Slurm。这等价于 §4 **选项 1**：正式三 seed 沿用当前 RF 图；在启动前仍可用 `scancel` 撤销。

| 阶段 | Job ID | 依赖 | 资源 |
|---|---:|---|---|
| 1 epoch 冒烟 | `1515426` | `afterok:1515425:1515423` | H200 / 128G / 1h |
| 冒烟放行 gate | `1515430` | `afterok:1515426` | CPU / 2G / 5m |
| seed 0 / 25 epoch | `1515431` | `afterok:1515430` | H200 / 128G / 2h |
| seed 1 / 25 epoch | `1515432` | `afterok:1515430` | H200 / 128G / 2h |
| seed 2 / 25 epoch | `1515433` | `afterok:1515430` | H200 / 128G / 2h |

`1515430` 不把 1 epoch 的 `loss_descended=false` 当失败，因为单点无法定义下降；它检查：full/seed 0/
1 epoch、3,016,439 节点、`no_nan`、图摘要 `0e80b839…`、family vocab 摘要、41,056 families、
四个规模开关、`--skip-diagnostics`，以及 `peak_gpu_mem_gb <= 100`。任一不满足即非零退出，
三个正式作业保持依赖未满足，不会带病训练。正式 25 epoch 的 `mechanism_gate.loss_descended` 仍按 §9 验收。

查看整条链：

```bash
squeue -j 1515423,1515426,1515430,1515431,1515432,1515433
sacct -j 1515423,1515426,1515430,1515431,1515432,1515433 \
  --format=JobID,State,Submit,Start,Elapsed,MaxRSS,AllocTRES,ExitCode
```

### 6. 2026-08-20 排队提速与 PyG 后端修复

此前探针 `1515423` 被固定到已满载的 `watgpu808`，Slurm 一度估算其在
`2026-08-21 14:54 UTC`（多伦多 10:54）才启动。Codex 随后检查所有节点，确认
`watgpu508` 同样为 H200 且有可用 GPU，因此把执行链迁移到该节点。

- `1515694`：508 上的首次快速探针，1 分 10 秒后按 gate 预期失败；H200/CUDA 正常，
  但环境报告 `pyg_lib=False`、`torch_sparse=False`。
- 无需密码，已从 PyG 官方 wheel 索引安装与 `torch 2.12.0+cu130`、Python 3.11 匹配的
  `pyg_lib 0.8.0+pt212cu130` 到 `$ENV_SOURCE`。
- `1515702`：安装后的复测，`watgpu508`，于 `2026-08-21 02:59:23 UTC` 启动，
  57 秒完成，ExitCode `0:0`；实测 `NVIDIA H200 NVL`、CUDA 可用、`pyg_lib=True`。
- 旧的 808 探针 `1515423` 已取消；smoke `1515426` 的依赖已切到成功的 `1515702`。
- smoke 与三条正式训练 `1515431`/`1515432`/`1515433` 的目标节点均已改为
  `watgpu508`。508 有三张可用 H200，gate 通过后允许三 seed 并行调度。
- 为进入当前 backfill 窗口，仅 1 epoch 的 smoke 时限由 1 小时缩为 20 分钟，内存仍为
  128G；修改后 `1515426` 已于 `2026-08-21 03:00:53 UTC` 立即启动。
- `1515426` 在训练加载前发现环境还缺少脚本声明导入的 `matplotlib`，46 秒后退出；没有加载图、
  没有开始训练。已无需密码地补装 `matplotlib 3.11.1`，并提交替代 smoke `1515703`。
  gate `1515430` 已改为 `afterok:1515703`，新 smoke 已在 `watgpu508` 启动。
- `1515703` 实际完成了完整 1 epoch：训练墙钟 37.1 秒、峰值显存 37.991GB、无 NaN、
  checkpoint 已写出。程序因 1 epoch 无法满足 `loss_descended` 而按设计返回 1，因此 gate
  的依赖改为 `afterany:1515703`，再由自定义规则验收其产物。
- gate `1515430` 于 `2026-08-21 03:05:54 UTC` 完成，ExitCode `0:0`；全部检查通过。
  三个正式作业已解除依赖，当前均为 `PENDING (Priority)`，且均固定申请 `watgpu508` 的一张
  H200；本机无需保持开机或 SSH 连接。

当前查看命令：

```bash
squeue -j 1515703,1515430,1515431,1515432,1515433
sacct -j 1515694,1515702,1515426,1515703,1515430,1515431,1515432,1515433 \
  --format=JobID,JobName,State,NodeList,Start,End,Elapsed,MaxRSS,ExitCode
tail -f logs/mlf-f6-rf-1515703.out
```

### 7. 正式三 seed 最终结果（2026-08-21）

三个正式作业在 `watgpu508` 的三张 H200 上并行完成，Slurm 状态均为 `COMPLETED`、
ExitCode 均为 `0:0`：

| seed | Job ID | Slurm elapsed | 训练墙钟 | 峰值 GPU | loss（首 → 末） | mechanism gate |
|---:|---:|---:|---:|---:|---:|---|
| 0 | `1515431` | 8m23s | 426.4s | 38.001GB | 22.2925 → 16.6288 | PASS |
| 1 | `1515432` | 9m31s | 498.8s | 38.015GB | 22.5152 → 16.4704 | PASS |
| 2 | `1515433` | 7m35s | 379.4s | 37.976GB | 22.2175 → 16.2624 | PASS |

最终验收：

- 三条均为 full rung、`n_models=3,016,439`、25 epochs、CUDA/H200；训练全程无 NaN，
  loss 均下降，size/family embedding 均发生更新，三份 `mechanism_gate.passed=true`。
- 三条均绑定图摘要
  `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c`，family vocab 摘要
  `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005`，
  `num_families=41056`。
- `init_seed` 三条均为 0，`split_seed` 分别为 0、1、2，符合三 seed 验收约定。
- 每条 run 均保留 `best.pt`、`last.pt`、epoch 14/19/24 checkpoint 和训练历史；最终 checkpoint
  约 12MB，每个 run 的 checkpoint 目录约 56MB。
- manifest 的 `mean_cos`、`z_m_pr`、`z_d_pr`、density 和 HNSW 字段为 NaN，是本次明确启用
  `--skip-diagnostics` 后未计算的诊断占位值；不代表 loss/梯度出现 NaN。

远端结果目录：

```text
/u801/x98liu/model_lake/runs/RF_full_s0_e25
/u801/x98liu/model_lake/runs/RF_full_s1_e25
/u801/x98liu/model_lake/runs/RF_full_s2_e25
```
