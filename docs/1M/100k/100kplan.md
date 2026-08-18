# R2 = 100K 档执行手册（Runbook）

上级规划：[`plan.md`](plan.md)，其中定义了五档梯子 R0–R4、双层湖设计与 S0–S5 阶段划分。

本档是梯子的第一个真实放大档，也是 R3/R4 全部基础设施的首次端到端验证。100K 跑通之后，
500K/1M 主要是更换 `--limit` 与资源申请。

目标规模 N = 100,000 模型，其中 CORE 30,183、HALO 69,817。
算力为 watGPU 单卡。实测分配到的节点是 `watgpu208`，NVIDIA RTX 6000 Ada Generation，48 GB 显存，
不是计划最初假设的 A100/H100；显存超过本档需要的 24 GB，但单 epoch 时间需按该卡实测，不能沿用别处的数字。

---

## §0 交付物与当前状态

### 0.1 这一档要交付什么

| # | 交付物 | 验收 |
|---|---|---|
| D1 | `hgraph_100k.pt`（100,000 模型 / 9,603 数据集节点 / CORE 监督冻结） | §12 的 G-B 系列全过 |
| D2 | R0(12K) 与 R1(30K) 的锚点复现数字 | 12K `gold@10` 与历史基线一致，判据见 G-C2 |
| D3 | R2 三轴数字（A 精度 / B iso-recall 延迟 / C 冷启动分层） | §12 的 G-D 系列全过 |
| D4 | `z_m [100000,128]` + `hnsw_100k.bin` + `MANIFEST.json` | 行序断言通过 |
| D5 | `docs/1M/S4/R2_EXECUTION.md`，包含失败过程留痕，沿用 P0–P5 的体例 | — |
| D6 | 可直接复用于 R3/R4 的爬取、建图、训练、评测代码 | R3 只需更换规模参数 |

明确不在本档范围内，以防范围蔓延：ModelLens 的对照实测（D-16 定为外推）、HALO 标签进入监督
（D-10 定为不进）、500K 与 1M 的任何数字。

### 0.2 当前状态（截至 2026-08-17）

已完成的部分：

- T2 的候选发现与均衡选样已完成并冻结（2026-08-03，本地），选出 69,817 条 HALO，20 条总体门过 18 条。
- T3 的梯子已落盘，100,000 行，五条出闸门全过，`ladder_sha256 = ccf288aee2a3caee…`。
- T4 的特征已在 watGPU 上产出（`x_m.npy`，sha256 `d7573a02…`），五条出闸门过四条；
  未过的是 G-B4（CORE/HALO 可分性 AUC 实测 0.9729，原门要求 ≤0.75），已按 D-39 改判为双口径并归因。
- T5 的 100K 图已建成并通过 33 条门禁；因血缘 `relation_id` 缺失，图已按 D-44 重建为 `16f35214…`。
- T6 的训练已跑完，机制门 12/12 通过，R1（全 CORE 30K）首次跑通，D-9 解除。
- T7 的导出与索引、T8 的 A/B/C/D 四轴评测均已产出数字，见 §11。
- T0 要求的 D-14 A/B 对拍表已由 [`T7.5.md`](T7.5.md) 以扩展形式补做（三档 × 三臂 × 三 seed）。

尚未解决的部分：

- G-C2 锚点门不通过。同一个 `split_seed=0`、同一批 517 个查询，P3 存档为 0.4159，本次为 0.3868。
  T7.5 的 18 个 run 已排除对比损失采样与扇出上限这两个近似作为原因，差异归到环境与采样后端；
  是否把该门从「复现点值」改写为「3-seed 区间覆盖历史值」需要单独裁定，见 §12 与 §16。
- G-A1（T0 的 12K 锚点门）在 T0 阶段从未执行，第一次真正跑到是在 T7，且未通过。
- G-D5 被记为违反：单 seed 比较下 100K 的 `gold@10` 高于 30K。按 T7.5 的三 seed 均值方向恢复正常，
  该记录是否解除需按新口径重新裁定。
- G-E1 是 T8 实测新发现的问题：`cold` 层的塌缩程度高于 `frozen` 层。属于新实验，不在 R2 范围内，
  但要写进 `R2_EXECUTION.md`。
- T1 的部分远端记录项仍待在 watGPU 上补齐。

---

## §1 三条实验有效性约束

以下三条是本档结论成立的前提条件，违反任何一条，这一档的数字都不能与 R0/R1 比较。
在 [`T0.md`](T0.md)、[`T3.md`](T3.md)、[`T7.md`](T7.md)、[`T6more.md`](T6more.md) 等执行记录里，
它们被称为「铁律 1」「铁律 2」「铁律 3」，编号一一对应。

### 约束 1：CORE 逐字节冻结

CORE 的 30,183 行特征、size 与 family id、`trained_on` 边、`unique_model_id` 表，
都从 `hgraph_ml_v2.pt` 原样复制，不重新计算，也不重新嵌入。

原因是 MiniLM 在不同设备与不同 batch 组成下末位存在浮点差异。重新嵌入一次，CORE 就不再是
P3/P5 使用的那个 CORE，R0–R4 之间「N 是唯一变量」的比较不再成立，历史战报也无法与本档对账。

建图末尾应有以下断言：

```python
core = torch.load("hgraph_ml_v2.pt", weights_only=False)
assert torch.equal(data["model"].x[:30183], core["data"]["model"].x)
assert torch.equal(data["model"].size_bucket_id[:30183], core["data"]["model"].size_bucket_id)
assert torch.equal(data["model"].family_id[:30183],     core["data"]["model"].family_id)
assert umi.iloc[:30183]["model"].tolist() == core["unique_model_id"].sort_values("mappedID")["model"].tolist()
assert torch.equal(data["dataset"].x, core["data"]["dataset"].x)          # 数据集侧完全不动
assert torch.equal(data[TRAINED_ON].edge_index, core["data"][TRAINED_ON].edge_index)
```

### 约束 2：HALO 使用与 CORE 相同的描述符函数

仓库里有两个同名不同义的函数，需要区分：

| 函数 | 位置 | 产出文本 |
|---|---|---|
| CORE 使用的 | [`scale/modellens_build_graph.py:67`](../../scale/modellens_build_graph.py#L67) `model_descriptor(mid, family, size_b)` | `"bert base uncased family bert 0.11B params"` |
| 不用于 HALO | [`stage1BuildTransferGraph/d0_build_graph.py:102`](../../stage1BuildTransferGraph/d0_build_graph.py#L102) `model_descriptor(d, mid)` | `"bert base uncased fill-mask transformers pytorch en datasets: bookcorpus wikipedia"` |

如果 HALO 用了 HF 富字段版，两群模型的 `e_desc` 文本在长度、词表与句法上都会系统性不同，
MiniLM 空间里 CORE 与 HALO 因此天然可分。HALO 就很难挤掉 CORE 的 gold，`gold@10` 几乎不降，
得到的「扛住了 83 倍干扰」的结论就不成立。

做法是从 HF 元数据里只抽 `family` 与 `size_b` 两个量，喂进 CORE 的 `model_descriptor`。
HF 的 tags、pipeline_tag、library_name 除用于推断 family 外一律不用（D-18）。
副作用要在报告里写明：我们主动放弃了 HF 上更丰富的模型元数据，这是受控实验的要求，不是能力上限。

同一个函数不等于同样的信息量。`model_descriptor` 在 `size_b` 为 NaN 时会整段省略 `"<x>B params"` 子句，
在 family 为空时会省略 `family <x>` 子句，因此字段覆盖率差异本身就是一条文本差异通道，
约束 2 的字面表述管不到它。实测的覆盖率差异见 §7.3。
这一层的守门员是 G-B4 的可分性 AUC；实测 0.9729，未落在原定的 [0.5, 0.75] 区间内，
已按 D-39 改判为「绝对可分性 + 流水线增量」双口径报告，并按 D-26/D-40 完成归因。

### 约束 3：评测时候选池必须是 100,000

最可能出现的静默错误是评测代码只对 CORE 的 30,183 个模型打分，`gold@10` 因此纹丝不动。

```python
# five_metric_eval / global_metrics 里应有
assert z_m.shape[0] == args.expect_n, f"candidate pool is {z_m.shape[0]}, expected {args.expect_n}"
```

并在 `MANIFEST.json` 里回写实际的 `N_candidates`。§12 的 G-D1 检查这一条。

---

## §2 阶段依赖

```
        ┌─ T0 代码改造（本地，12K 上验收）──────────┐
        │                                            ▼
        ├─ T2 HF 爬取 ──> T3 梯子 ──> T4 特征 ──> T5 建图 ──┐
        │  （本地即可，产物 25 MB 可移植）                   ▼
T1 watGPU 环境 ───────────────────────────────> T6 训练 ──> T7 导出+索引 ──> T8 评测
```

关键路径是 T1 与 (T2 → T3 → T4 → T5) 并行，然后 T6 → T7 → T8。
T0 与 T2 都不依赖 T1，因此这两项可以最先开工。

T2 不依赖 watGPU 的外网访问。实测其产物只有 24.8 MB、墙钟 46.8 秒，本地跑完上传是秒级操作，
外网分叉（D-12）保留为 R4 的优化项（D-29）。

按计划，T6 应在 T0 出闸门全绿之后才开始。实际执行中这一顺序没有被遵守，后果记在 §12 的 G-A1。

---

## §3 T0：代码改造（本地，在 12K 图上验收）

100K 档需要 `plan.md §4` 十项改造中的 7 项。第 5、9、10 项在 100K 上用不到，
因为数据集数不变、fp32 装得下、图文件约 250 MB，这三项留到 R3/R4（D-21）。

### T0.1 建立基线

改造前先钉住对照组：

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only v2      # 应为 22/22 全绿
# 记录当前 12K 基线（若 P3 产物还在可直接引用，否则重跑一次）
Set-Location D:\research\model_lake\codes
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.export_ours `
  --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt `
  --seed 0 --epochs 25 --tag R0_baseline_pre
```

记下 `gold@10`，历史存档值是 0.4159，作为后续每一项改造的锚点。
需要注意这个值是 `--seed 0` 的单次结果，其噪声带此前从未测过（D-43），
T7.5 在 12K `exact` 臂上测到的三 seed 散布是 0.3395–0.4681，说明单次值本身方差很大。

### T0.2 改造项

按以下顺序改，每项改完立刻跑一次锚点。

**第 1 项：扇出上限 loader，`stage2TrainGraphSAGE/sampling.py`。**
改动位置是 `LightLinkLoader._khop_closure`（[sampling.py:88](../../stage2TrainGraphSAGE/sampling.py#L88)）。
当前实现做全邻域闭包，并用 `torch.isin(src, tensor(sorted(cur)))` 全表扫描。
改为 CSR 支撑的有界扇出采样：

```python
def __init__(self, ..., num_neighbors=(10, 10), gen=None):
    self.fanout = list(num_neighbors)
    # 预建 CSR：按 src 排序 + ptr，邻居查找从 O(E) 降到 O(log N + deg)
    self._csr = {et: _build_csr(ei, n_src) for et, ei in self._edges.items()}

def _khop_closure(self, seed_m, seed_d):
    # 每跳：对每个 frontier 节点，从其邻居里随机抽 min(deg, fanout[hop]) 个
```

参数取 `num_neighbors=(10,10)`，血缘关系用更大的扇出 `(20,20)`，
因为血缘边是冷启动模型唯一的边，CLAUDE.md Step 4 对它规定了更低的丢弃率，这里对应更高的保留率。
验收方式是单测 `test_fanout_bounded`（fanout=(2,2)、batch=128 时子图模型节点数不超过 128×(1+2+4) 的可证上界），
以及 12K 上 `gold@10` 与 T0.1 基线在 3 seeds 内统计不可区分。

**第 2 项：稀疏 `M`，`stage2TrainGraphSAGE/losses.py`。**
涉及 [`topk_membership`:415](../../stage2TrainGraphSAGE/losses.py#L415)、
[`pool_membership_by_root`:426](../../stage2TrainGraphSAGE/losses.py#L426)、
[`global_positive_density`:467](../../stage2TrainGraphSAGE/losses.py#L467) 以及
`dataset_to_model_contrastive` 里的 `M[cand, d]`。
100K × 9,603 的稠密矩阵是 3.84 GB，1M 时是 38.4 GB，而非零元只有约 31K，稠密度 3×10⁻⁵。
把 `M` 改为 `torch.sparse_csc_tensor`，`pool_membership_by_root` 的 python 循环换成按 root 的 `scatter_max` 聚合。
验收是 `test_sparse_M_equivalence`：12K 上稀疏版 `.to_dense()` 与原稠密版逐元素相等，
`pool_membership_by_root` 输出相同。

**第 3 项：对比损失采样负例，`losses.py:498` 的 `contrastive_loss`。**
即使扇出上限把子图压到 20K 节点，`[20000,20000] × 4 张 × 4B` 仍是 6.4 GB。
A 方案是每锚点抽 `n_neg` 个负例（默认 256，与 L1L3b 的 `global_n_neg` 同源），复杂度降到 O(B×n_neg)。
这一项改动的是既有的最优配置，D-14 因此要求做 A/B 对拍并写进报告：

| 配置 | seeds | gold@10 mean ± bootstrap CI |
|---|---|---|
| 原版全 N² 对比损失 | 0,1,2 | 基线 |
| 采样负例 n_neg=256 | 0,1,2 | 见 T7.5 |
| 采样负例 n_neg=1024 | 0,1,2 | 未跑 |

若采样版显著更差，改用 B 方案（分块 logsumexp，数学等价，只省显存，时间仍是 O(N²)），
或把 100K 的子图上限压得更小，不接受未经记录的退化。

该对拍在 T0 阶段没有产出，后由 [`T7.5.md`](T7.5.md) 以扩展形式补做：
固定同图、同 `split_seed`、同 25 epoch、同 L1L3b 配置，在 12K/30K/100K 三档上各跑三个 seed，
比较 `approx`（n_neg=256）与 `exact`（全 N²）两臂。
结果是 `Δ = exact − approx` 为 +0.0194 / +0.0122 / +0.0069，随 N 单调收窄，三档都在 1 个 seed 标准差以内；
而 `exact` 在 100K 上的墙钟代价是 25.6 倍。结论是采用采样版，代价已量化并披露。
未跑 n_neg=1024 那一档，因为 256 并未表现出显著劣化。

**第 4 项：分块推理，`scale/export_ours.py:122,135`。**
当前实现是 `model(data.clone().to(device))`，整图搬 GPU。
100K 上 `x[100K,448]` 是 179 MB，2 层 hidden 乘 5 种关系的中间量使峰值预估到 4–8 GB，
本档的卡装得下，但 R3/R4 装不下，因此在 100K 上先做首次实战验证。
改法是预分配 `z_m = torch.empty(N,128)`，用扇出 loader 按 `batch_size=50_000` 遍历模型节点，
在 `torch.no_grad()` 下逐块写入。
写入必须用 `z_m[batch['model'].n_id] = out`，不能用 `z_m[i*B:(i+1)*B] = out`；
CLAUDE.md 把行序错位列为三大风险之首，这一行是它最容易发生的地方。
验收是 12K 上分块推理结果与全图前向逐元素 `max|Δ| < 1e-5`。T7 实测该值为 0.00e+00。

**第 5 项：流式打分，`top1_eval.py:68` 与 `scale/global_metrics.py`。**
当前 [export_ours.py:128](../../scale/export_ours.py#L128) 把 517 个查询 × N 的完整分数向量物化进一个 dict，
100K 上是 207 MB，尚能承受，但 1M 时是 2 GB 且 CPU matmul 很慢。
改法是把打分搬到 GPU，`z_m` 常驻显存，逐查询或分块计算 `z_d_q @ z_m.T`，
只保留 gold rank、gap rank 与 top-K id。
验收是 P3 验证过的「`five_metric_eval.gold@10 == global_metrics.gold@10` 逐查询对账」改完后重跑仍然相等。

**第 6 项：诊断可关，`losses.py:467`。**
增加 `--skip-diagnostics`。100K 档默认开着，因为稀疏化后它不再是瓶颈；R3/R4 默认关闭。

**第 7 项：HNSW 多线程、落盘与 iso-recall，`export_ours.py:58`。**
`init_index(max_elements=N)` 已参数化，新增 `--hnsw-threads`、`--ef-construction`、`--M`，
并用 `idx.save_index()` 落盘以避免评测时重建。
新增 `tune_ef_for_recall()`，二分 `ef_search` 使 `recall@50 ≥ 0.99`，返回最小可行 ef，
延迟必须在这个 ef 下测量（§11.2）。

### T0.3 T0 出闸门

- [ ] 7 项全部改完，各自单测通过
- [ ] 锚点门：12K 上 3 seeds，`gold@10` 与 T0.1 基线 bootstrap CI 重叠（= G-A1）
- [x] 分块推理与全图前向 `max|Δ| < 1e-5`（T7 实测 0.00e+00）
- [x] `five_metric_eval == global_metrics` 逐查询对账仍成立（T0 实测 0/517 不一致，T7 七个 run 复验 PASS）
- [x] D-14 的 A/B 对拍表已产出（由 T7.5 补做，见上）

计划规定 T0 不过就不进入 watGPU 训练，理由是在未验证的代码上占用 GPU 时间没有收益。
实际执行没有遵守这一条，T6 在锚点门执行之前就跑完了，相关后果记在 §12 的 G-A1 与 G-C2。

---

## §4 T1：watGPU 环境

完整步骤见 `plan.md §2`，这里只列 100K 档的增量。
执行 `plan.md §S0.1` 的五组探测命令，结果写入 `docs/1M/S0/WATGPU_ENV.md`。

### 4.1 执行边界与集中配置

watGPU 登录节点仅承担 Git 操作、环境管理、轻量文件检查、Slurm 提交与监控。预处理、批量下载、模型加载、训练、评测及 GPU 检查均在 Slurm 分配的计算节点执行。交互式 allocation 仅用于短时调试和验证；正式运行统一使用 `sbatch`。本地 SSH 断开不影响已提交的 batch job。

路径和资源参数集中写入未含密钥的配置文件或作业环境。仓库仅提交占位模板 `configs/watgpu.example.env`；含账号、token 或实际路径的文件不进入 Git。

| 变量 | 含义 | T1 要求 |
|---|---|---|
| `PROJECT_ROOT` | 远端代码目录 | 登录后确认，不写入本地 Windows 绝对路径 |
| `DATA_ROOT` | 数据、图与只读快照目录 | 先确认容量、quota 与清理策略 |
| `OUTPUT_ROOT` | `runs/`、checkpoint、metrics 与日志目录 | 与代码目录解耦，保证可续跑 |
| `CACHE_ROOT` | Hugging Face、Torch 与 pip cache | 显式设置 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR` |
| `ENV_ACTIVATE` | venv/conda 激活命令 | 以 `plan.md §S0.2` 的实测结果为准 |
| `PARTITION` / `SLURM_ACCOUNT` | 分区与 account | 由实时权限探测填写 |
| `GPU_CONSTRAINT` | 可选 GPU constraint | 默认留空；仅在实测表明确有必要且集群支持时填写 |

`$WORK` 仅在实时探测确认后作为上述根目录的来源，不预设 `/scratch`、`/project` 或其他大容量文件系统存在。

### 4.2 一次性接入与首次登录探测

本地设备使用独立的 Ed25519 SSH key。公钥上传 AuthMan，私钥保留在本机并设置 passphrase；管理员确认账号启用后再诊断登录问题。首次连接须核对 host fingerprint，不关闭 host-key verification。SSH 配置使用 `<UW_USER>`、`<PATH_TO_PRIVATE_KEY>` 等占位符，不在计划或仓库记录真实私钥路径。

首次登录仅执行轻量探测：

```bash
whoami
hostname
pwd
df -h
du -sh "$HOME" 2>/dev/null || true
quota -s 2>/dev/null || true
conda --version 2>/dev/null || true
python --version
git --version
sresources
sinfo
squeue -u "$USER"
```

`WATGPU_ENV.md` 需记录 `HOME_DIRECTORY`、`PROJECT_DIRECTORY`、`DATA_DIRECTORY`、`OUTPUT_DIRECTORY`、home quota、大文件目录、授权 partition/account、可用 GPU constraint、Python 与环境管理器版本。存储位置或配额未确认时，暂停大数据上传。GPU 型号、driver、CUDA runtime 与显存只在计算节点 allocation 内记录。

### 4.3 代码、数据与环境同步

- 代码默认通过 private Git repository 同步。每次正式运行前执行 `git pull --ff-only`，记录 `git rev-parse HEAD` 与 `git status --short`。生产作业通常使用 clean commit；有意保留的未提交变更写入运行目录的 `metadata/uncommitted.patch`。
- 大文件重复传输优先使用 `rsync --partial`；只追加数据可使用 `--append-verify`。`--delete` 不进入默认命令。CORE 图、corpus 与关键分片在源端和远端校验 SHA-256。
- 远端环境由锁定的 requirement/environment specification 重建。PyTorch/CUDA 组合依据计算节点 driver 与官方兼容性确定，不直接复制本地 Windows wheel 选择。安装后保存 `pip freeze` 或等价环境记录，并执行 `python -m pip check`。
- dataset、checkpoint、cache、log、token、SSH key 与私有 `.env` 不进入 Git。HF token 通过作业环境注入，不写入脚本和 Slurm 日志。
- 数据上传前确认 privacy、license、ethics 与 residency 限制。受限数据放入非共享目录，并使用 `umask 077` 或等价权限控制。

实测得到的两条环境事实需要在作业脚本里处理：`$HOME` 挂了 noexec，venv 不能直接在 `$HOME` 里执行，
作业脚本把整个环境 `cp -a` 到计算节点的 `/tmp` 再运行，每次作业固定增加约 46.5 秒；
远端环境最初缺 `pyg-lib`（或 `torch-sparse`/`torch-scatter`）、`hnswlib`、`faiss`，
T4 用不上因而没有暴露，这三个包是 T6/T7 开工前要先确认的依赖。

### 4.4 交互式 GPU 冒烟

短时 allocation 用于框架与最小项目链路验证：

```bash
salloc --partition=<PARTITION> --gres=gpu:1 \
  --cpus-per-task=4 --mem=16G --time=00:30:00
# 按 watGPU 返回的主机名进入已分配计算节点，再执行：
<ENV_ACTIVATE_COMMAND>
hostname
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("smoke checksum:", float((x @ x).mean()))
PY
```

项目冒烟需完成 import、一个 tiny batch 的 forward/backward、checkpoint save/reload/resume、一个既有评测指标及输出路径检查。仓库和 resolved config 需确认不存在 `C:\` 或 `D:\` 路径。验证结束后退出计算节点并释放 allocation。

### 4.5 100K 档的资源需求

| 项 | 需求 | 说明 |
|---|---|---|
| GPU | 1 卡，显存 ≥ 24 GB | 实测分到 RTX 6000 Ada 48 GB；训练峰值 12K 0.583 GB、100K 1.737 GB |
| `--mem` | 96G | 峰值出现在建 HNSW 与稀疏 M 构造 |
| `--cpus-per-task` | 8 | HNSW 建索引多线程 |
| `--time` | 00:30:00 | 按实测修正（D-45），最慢一次 2m44s，含 env staging |
| 磁盘 | 约 1 GB | 爬取 0.025 GB + 特征 0.2 GB + 图 0.25 GB + 索引 0.1 GB + 运行目录 |

`--time` 的取值按实测而不是留大额占位值。Slurm 按申请的墙钟排队，4 小时的占位值曾让六个两分钟的作业
在一张 GPU 后被排成每 4 小时一个，改为 00:20:00 后压缩进同一个晚上，节省约 14 小时。

外网探测仍然照做并记入 `WATGPU_ENV.md`，但结论两条路都可行（D-29）：

```bash
srun --partition=<P> --gres=gpu:1 --time=00:05:00 --pty bash -c \
  'curl -sS -m 10 -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models?limit=1'
```

返回 200 表示 T2 可以在 watGPU 上跑，这对 R4 的约 200 MB 爬取量有意义；超时或拒绝则在本地跑并上传。
R2 的 T2 已在本地完成，不需要在 watGPU 重跑。

### 4.6 Batch 提交、监控与结果回收

提交 wrapper 在调用 `sbatch` 前创建 `logs/`；`scripts/watgpu/train_rung.sh` 启动后创建运行目录，启用 `set -euo pipefail`，并通过 `srun` 调用 T6 的原始训练命令。每次实验分配唯一的 logical `RUN_ID`；Slurm job ID 单独记录，requeue 或续跑沿用原 `RUN_ID`。

运行元数据至少包含：时间戳与时区、hostname、Git commit 与工作树状态、exact command、resolved config、seed、数据 manifest/split、Python 与依赖版本、framework/CUDA/GPU/driver、partition/account、CPU/RAM/GPU/time request、resume checkpoint、exit code、Slurm final state、最终 metrics 与 artifact paths。`nvidia-smi` 和环境快照在计算节点写入 `${OUTPUT_ROOT}/runs/${RUN_ID}/metadata/`。

```bash
cd <REMOTE_PROJECT_ROOT>
mkdir -p logs
sbatch --export=ALL,RUN_ID=<descriptive_run_id> scripts/watgpu/train_rung.sh

squeue -u "$USER"
scontrol show job -dd <JOB_ID>
tail -f logs/<job-name>-<JOB_ID>.out
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

监控采用人工检查或低频、有界轮询。`COMPLETED` 仅表示 Slurm 进程结束；科学验收还需确认最终 checkpoint、可解析 metrics、无 NaN/Inf、T8 完成、commit 与 config 对应正确及所有分片齐全。

结果回收优先下载 metrics、resolved config、metadata、logs、best/final checkpoint 与报告所需表格。关键产物校验文件大小和 checksum；本地验证完成前保留远端副本。

Windows PowerShell：

```powershell
scp -r watgpu:"<REMOTE_OUTPUT_ROOT>/runs/<RUN_ID>" "<LOCAL_RESULTS_PATH>"
```

WSL/Linux/macOS：

```bash
rsync -avh --partial --info=progress2 \
  watgpu:"<REMOTE_OUTPUT_ROOT>/runs/<RUN_ID>/" \
  "<LOCAL_RESULTS_PATH>/<RUN_ID>/"
```

### 4.7 T1 基础设施出闸门

- [ ] AuthMan 公钥、账号确认与 SSH 登录通过；私钥未上传
- [ ] home quota、大文件目录、partition/account 与输出保留策略已确认
- [ ] 远端环境安装完成，`pip check` 与计算节点 CUDA smoke test 通过
- [ ] 数据 manifest、代表性 checksum 与 CORE corpus 审计通过；无本地绝对路径
- [ ] tiny batch forward/backward、既有 metric 与 checkpoint save/reload/resume 通过
- [ ] tiny `sbatch` 正常结束，日志、metadata 与返回的 Slurm job ID 已落盘

---

## §5 T2：HF 候选发现与均衡 HALO 构建（已完成，2026-08-03，本地）

执行记录见 [`T2.md`](T2.md)。本节的数字都是实测值。

### 5.1 HALO 总体的定义

HALO 不是「按 `downloads` 降序爬取的前 69,817 条」。早期版本（v1）按这个口径取前 15 万条，
技术上可行，但科学上不可用：

| 观测量 | v1 前 15 万 | v2 选出的 69,817 |
|---|---|---|
| 最大发布者占比 | 24.21%（`mradermacher`） | 0.902% |
| 有效发布者数（1/HHI） | 15.1 | 2,889.1 |
| `quantized-conversion` 占比 | 61.98% | 第三方 14.999% + 官方 2.585% |
| `text-generation-chat` 占比 | 50.77% | 18.000% |
| 任务分布归一化熵 | 0.601 | 0.918 |
| 元数据质量通过率 | 70.45% | 100.000% |

这是采样机制的产物，不是样本量问题。下载计数器奖励的是被自动化流水线反复拉取的次数，
再爬 100 万条只会让搬运仓库的绝对数变大。对下游的具体伤害（hub 过平滑、
对比损失的负样本实际上是同一个模型、跨任务检索无法测量、C 轴归因与 `source_type` 共线）见 `T2.md §1`。

因此 v2 把爬取与选样分成两级（D-30）：

```
多源候选发现  →  标注  →  家族/近重复归并  →  配额分配  →  确定性选样  →  缺口回补  →  冻结
   （允许有偏）                                        （不允许有偏）
```

v1 的 15 万条原样保留、未被覆盖，降级为审计件与候选源之一。

### 5.2 发现阶段

发现阶段跑 532 个查询（全局 5 序 + 56 任务 × 2–3 序 + 34 语种 × 1–2 序 + 30 生态 + 10 结构 + 110 缺口回补），
共 1,799,891 次原始命中，得到 537,025 条唯一候选（重叠 3.35 倍，64.7 MB，约 28 分钟）。
产物是可续、只追加的，R3/R4 从 `PLAN_STATE.json` 继续。

三条实测结论影响 R3/R4 的做法：

前缀嵌套只是近似，因为游标建在可变字段上。`Link: rel="next"` 里的游标 base64 解出来是
`{"$or":[{"downloads":X,"_id":{"$gt":...}},{"downloads":{"$lt":X}}]}`，keyset 建在 `downloads` 上，
而 `downloads` 每天在变。R3/R4 隔周续爬时，期间涨过游标位的模型会被跳过。
因此按规范化 id 去重是必需的（已实现，R2 本次 0 重复，但那是 46.8 秒窗口的结果，不是保证），
并且每一档的 `snapshot_date_utc` 分别记录、分别披露，报告里不写成「同一份 1M 快照的前缀」。

全局排序流之间的边际收益衰减很快：`global:downloads` 新增 100%，`global:likes` 51.8%，
`lastModified` 21.9%。覆盖来自按任务与语种切分，不是再加几种全局排序。

浅层语种查询几乎无用。各语种按下载量的前 240 条都是同一批多语大模型，新增为 0；
需要深挖（cap 6,000，按 `createdAt` 排序）才有增量。

### 5.3 `scale1m/hf_crawl.py`

请求必须走 `expand[]`，不能用 `full=True&cardData=True`：

| 路径 | 返回 `safetensors` | 返回 `siblings`（膨胀） | 后果 |
|---|---|---|---|
| `full=true&cardData=true` | 否 | 是（整个 repo 文件列表） | `size_b` 100% 缺失，全部 HALO 落 bucket 0，且不报错 |
| `expand[]=...` | 是 | 否 | 正确 |

`safetensors.total` 是 §6.1 里 `size_b` 的第一顺位来源，也是唯一可靠的来源。
走错这条路时 T3–T8 全程照常运行，只是 `e_size` 整群塌成一行，最后在 C 轴看到一个无法解释的结果。

```
GET https://huggingface.co/api/models?limit=1000&sort=downloads&direction=-1
    &expand[]=downloads&expand[]=likes&expand[]=pipeline_tag&expand[]=library_name
    &expand[]=tags&expand[]=createdAt&expand[]=safetensors&expand[]=cardData
```

实现要点：

- token 是可选的。限流为 `RateLimit-Policy: q=500;w=300`，即 500 请求 / 300 秒；
  `limit=1000` 是每页上限，15 万条只需 150 个请求，一个窗口都用不满。仍保留 `--token`/`$HF_TOKEN` 与主动退避。
- 断点续爬由 `CURSOR.json`（`next_url`、已写条数、分片内偏移与 stats）和 `SHARDS.json`（增量分片清单）支撑。
  进程在半行被杀时，续爬按 `CURSOR` 提交的条数截断，有单测覆盖。
- 429 与 5xx 走指数退避加 `Retry-After`，限流余量低于 25 时主动睡到窗口重置，`retry_reasons` 全部记账。
- 字段裁剪保留 `KEEP` 的七个顶层字段、`safetensors.total`，以及 `cardData` 的
  `base_model`/`datasets`/`model-index`；`model-index` 压成
  `(task, dataset, dataset_name, config, split, metrics[type,value])`，
  丢掉每条 metric 上数 KB 的 `verifyToken`。压缩前约 34 KB/条，gz 后 165 B/条。
- 爬虫不执行约束 2，`tags`/`pipeline_tag`/`library_name` 全量落盘。约束 2 是 T3/T4 的纪律。
  理由是快照不可重建：今天为纪律少存的字段，明天要用只能重爬一份不同日期的快照。
- 产物为 `hf_models_*.jsonl.gz`、`SHARDS.json`、`CURSOR.json`/`PLAN_STATE.json`、`PROVENANCE.json`，
  分片与 `PROVENANCE.json` 都设为 `chmod 444`。

v2 相对 v1 新增六个字段，都是选样与建图需要的：`author`（发布者集中度）、
`baseModels`（权威 parent 与 `relation`；`cardData.base_model` 只是手打字符串，没有关系类型）、
`config`（`architectures`/`model_type`）、`lastModified`、`gguf`（权威量化标志）、`gated/disabled/private`。
详见 `T2.md §3.2`。

另有两条爬取层面的事实需要记住。v1 的 `KEEP_CARD` 把 `cardData.language` 丢了，
该字段在 v1 的 15 万条里数量为 0，语种只能从 tags 的裸语种码猜，而裸 `^[a-z]{2,3}$` 会误收
`trl`/`sft`/`tf`/`jax`/`mms`；v2 补上该字段并改用 ISO-639 白名单，
所以 v1 的 40.88% 「语种未知」有一部分是爬取缺陷，报告里要一起说明。
另外 `library=` 参数被 HF 静默忽略（`library=timm` 返回全局下载榜首而不是 timm 模型），必须用 `filter=timm`；
它不报错也不返回空，只是不过滤。

### 5.4 标注与选样

```powershell
.\.venv\Scripts\python.exe -m scale1m.annotate_candidates  --candidates $C
.\.venv\Scripts\python.exe -m scale1m.select_balanced_halo --annotated $C\annotated.parquet --target 69817
```

标注（`taxonomy.py` 与 `annotate_candidates.py`）产出 18 类 supertask、5 类语种桶、7 类 source type、
量化方法与位宽、家族（血缘闭包，六级优先序）、近重复键与元数据质量分。
每个标签都带 `*_source` 与置信度，仓库名启发式一律自报来源，从不覆盖权威元数据。

选样（`select_balanced_halo.py`）用配额感知加权轮转，五轴同时跟踪，
再叠加发布者、搬运方、家族、近重复、top-10 五类硬上限。

关于分母的三条规则值得单独说明，它们都影响上限的含义：

所有百分比上限都是最终选出总体的份额，不是 `--target` 的份额（D-31）。
初版把 `text_gen ≤ 0.15 × requested_target` 写进选样过程，会产生两个问题：把 target 写大就能买到更宽的绝对上限；
跑不满 target 时，按实际产出算的份额已经越界。
修正做法是把上限按工作规模 `N` 参数化，二分搜索满足 `achieved(N) ≥ N` 的最大 `N`。
因为 `achieved` 随 N 单调不减、`achieved − N` 递减，判据单调，二分给出精确的可行最大值；
在该不动点上每个上限按构造就是实际产出的正确份额，出闸门再用真实计数独立复核一遍。
单测 `test_caps_hold_as_shares_of_the_actual_population_not_the_request` 用同一池分别请求 400 与 4000，
断言两者都满足各自 1.5% 的份额。

「英语侧 ≥ 75%」是关于语种构成的断言，分母是可归属语种的总体，即除 `language-neutral` 外的全部（D-32）。
把 ViT 计入英语份额既不测量任何东西，又与任务均衡冲突：语种无关的 5 个 supertask 在任务配额下合计就要超过 25%，
与「非英语侧 ≤ 25%」互斥。实测按全体作分母时均衡 HALO 卡在约 60,200 上不去，改用可归属分母后 69,817 一次探测即可行。

官方量化与第三方搬运分成两类 source type（D-33）。`Qwen/Qwen3-8B-GGUF`（第一方发布）与
`somebody/Qwen3-8B-i1-GGUF`（第三方重打包）不是同一类总体，判据是量化仓库作者是否等于 base model 作者。
15% 硬顶落在第三方类上，两者合计另设 20% 硬顶，防止用拆分抬高总量。

任务份额同时设目标与硬顶（D-34）：`max_share` 15% 是配额分配用的目标，
`hard_ceiling` 18% 是任何阶段不可越过的上限，G8 按实际总体检查。

### 5.5 T2 出闸门

v1 快照门（`raw/` 仍然只读、仍然可验，作为审计件）：

- [x] `PROVENANCE.json` 完整，`total_records = 150,000`；每片 sha256 复算一致（3/3）；分片只读（3/3）
- [x] 快照日期已记录，为 2026-08-03 (UTC)，v1 与 v2 同日
- [x] 规范化 id 去重后仍是 150,000（0 重复）；`downloads` 降序违例 0 条

v2 总体门 G1–G20（`verify_raw --selected`，全部按实际选出的 69,817 计算）：18 条硬门通过，
G14/G15 受联合约束所限并已定量披露。逐条见 `T2.md §6.5`，关键几条：

| ID | 判据 | 实测 |
|---|---|---|
| G3 | 普通最大发布者 ≤ 1.5% | 0.902%（v1 为 24.21%） |
| G4 | top-10 发布者 ≤ 10% | 4.045%（v1 为 40.42%） |
| G5 | 单个搬运发布者 ≤ 0.5% | 0.500% |
| G6 | 最大家族 ≤ 349 | 0.500%（v1 为 17.90%） |
| G7 | 第三方转换 ≤ 15%，全量化 ≤ 20% | 14.999% + 2.585%（v1 为 61.98%） |
| G8 | 无主任务 > 18%，目标 15% | 18.000%（v1 为 50.77%） |
| G10 | 英语侧 ≥ 可归属总体的 75% | 75.001% |
| G16 | 元数据质量通过率 ≥ 95% | 100.000%（v1 为 70.45%） |
| G18 | 选出数 = 可行最大值 | 69,817 = 69,817 = requested |
| G19 | 确定性复跑 id 相同 | 逐条相同 |

有效发布者数从 15.1 升到 2,889.1，有效家族数从 31.1 升到 3,197.7，任务分布归一化熵从 0.601 升到 0.918。
完整三方表见 [`T2_runs/distribution_report_v2.md`](T2_runs/distribution_report_v2.md)。

`verify_raw --forecast` 段不是闸门，它把 T3/T5/T8 能提前算的量先写死，防止事后重新叙述。

---

## §6 T3：canonical 化与 100K 梯子构建（已完成，2026-08-03，本地 CPU）

执行记录见 [`T3.md`](T3.md)。梯子 100,000 行已落盘，五条出闸门全过，
`ladder_sha256 = ccf288aee2a3caee…`。T3 不需要 GPU。

### 6.1 `scale1m/hf_canonicalize.py`

对每条 HF 记录产出建图需要的四个量，只要这四个：

| 量 | 来源 | 规则 |
|---|---|---|
| `unique_model_id` | `id` | 规范化 `strip().lower()`，用于与 CORE 去重，与 `scale1m.verify_raw.normalize` 是同一个函数 |
| `size_b`（十亿参数） | 只用 `safetensors.total`（D-26） | 缺失时为 `NaN`，`param_count_to_size_bucket(None)` 得到 bucket 0（unknown） |
| `family` | 四级解析，目标是 CORE 的命名空间（D-36） | 不在 vocab 且计数低于 `FAMILY_MIN_COUNT` 时归为 `Other`（id 0） |
| `lineage_base` | `baseModels` 优先，`cardData.base_model` 兜底（D-37） | 规范化后备用，`relation` 一并保留 |

family 的解析规则需要说明来由。真正可用的规则函数是
`dataset_embed/utils/fetch_metadata.py::_infer_one_family`，它只看名字，没有 tags 兜底，
并且与 CORE 不在同一个命名空间。CORE 的家族串来自 ModelLens 的 `model_profile.family`，
实测取值是 `bert`(6,909)、`llama`(1,492)、`vit`(1,308)、`xlm`(1,290)、`qwen`(925)，
即 HF `config.model_type` 的取值；而规则表返回 `BERT`、`LLaMA`、`ViT`、`Qwen`。
`family_vocab`（341 行）里两套都在，`KNOWN_FAMILIES` 先种子化，ModelLens 的小写串后动态录入。
直接用规则输出会让 HALO 的每个 llama 拿到 `LLaMA` 行、CORE 的每个 llama 留在 `llama` 行，
同一架构占两行嵌入、零共享，而且不报错。

| 策略 | 落进 CORE 真正用过的 207 个家族 |
|---|---|
| `_infer_one_family` 原样 | 0.64% |
| 仅小写化 | 33.98% |
| 仅 `config.model_type` | 29.63% |
| 四级解析（采用） | 48.47% |

四级解析的顺序是：`config.model_type` 且 CORE 用过则用它；小写化的规则输出且 CORE 用过则用它；
`config.model_type` 非空则用它（新行，但命名空间正确）；最后才用小写化的规则输出。每行记 `family_source`。

`lineage_base` 以 `baseModels` 为准。537K 候选上实测 `baseModels`（HF 结构化，带 `relation`）命中 234,445 条，
`cardData.base_model`（作者手打字符串）只有 12,713 条，只用后者会丢掉 94.9% 的血缘声明。
保留 `relation`（quantized/adapter/finetune/merge）是因为它正是 CLAUDE.md 里 `r_mm'` 离散有序权重需要的量，
T5 不必再从名字推断。

`size_b` 不加名字正则兜底（D-26）。`model_descriptor` 在 `size_b=NaN` 时整段省略 `"<x>B params"` 子句，
所以「多大比例带 size 子句」本身就是一条系统性文本差异，与用哪个描述符函数无关。三个选项的实测覆盖率：

| 群 | 带 size 子句 |
|---|---|
| CORE（`size_bucket_id ≠ 0`） | 47.91% |
| HALO，仅 `safetensors.total` | 见 §6.3 的 v2 实测 57.75% |
| HALO，加名字正则（`-7b-`/`-350M-`） | 约 78.60% |

名字正则能从 97,494 条缺失里救回 62,813 条（64.4%，主要是没有 safetensors 权重的 GGUF 量化仓库），
但救得越多离 CORE 越远。D-18 的原则「信息量对齐优先于信息量最大化」在这里落成数字，
主动放弃的这 62,813 条可从名字恢复的参数量要在报告里写明。

同时做分层打标，`layer` 列仅用于审计，不进训练（D-10）：

| layer | 判据 | T3 实测（v2 HALO） |
|---|---|---|
| `labeled` | 有可解析的 `model-index` 结果 | 6,619（9.48%） |
| `lineage` | 有 `base_model`，无 model-index | 25,268（36.19%） |
| `plain` | 都没有 | 37,930（54.33%） |
| `dropped` | 无 pipeline_tag 且无 tags 且无参数量 | 0，恒空 |

`labeled` 比 v1 口径高约 4 倍，因为质量门偏好有结构化元数据的模型，
这直接把 §11.1 的 `displacement_quality` 样本从 1,642 提到 6,619。
`lineage` 比 v1 口径低约四成，因为近重复上限削掉了同根下的密集派生，而那正是 `base_model` 声明最密的一群。
`plain` 过半（54.33%）要写进报告：超过一半的 HALO 既无标注也无可解析血缘，只能靠自身特征。

`dropped` 的判据恒为假（D-27）。实测 `tags` 覆盖率 100.00%，HF 给每个仓库自动挂
`region:us`、`license:*`、架构名之类的标签，所以那个合取式永远不成立。
判据保留不动，因为 `layer` 只进审计不进训练，恒空不产生下游影响，而改判据等于引入一条未验证的新过滤规则；
但报告里要写明它恒为 0，避免读者以为过滤生效了。

```bash
python -m scale1m.hf_canonicalize --candidates $C --core .../hgraph_ml_v2.pt
# 产出 canon/hf_canon.parquet（model, id_norm, size_b, family, lineage_base, layer,
#   downloads, rank + size_source / family_source / lineage_source / family_in_core_used）
```

输入是 `candidates_v2/`，不是 v1 的 `raw/`。

### 6.2 `scale1m/build_ladder.py`

梯子的主体是断言。行序错位不会抛异常，只会改写「哪个模型拥有哪一行嵌入」，
因此 concat 之前四条、之后四条，共八条：

```python
core_ids = set(normalize(m) for m in core_umi["model"])          # 30,183
halo_100k = pd.read_parquet(SELECTED_HALO)                       # 69,817，已均衡、已过门

assert len(halo) == n - n_core                     # HALO 规模
assert not (core_id_set & set(halo.id_norm))       # CORE ∩ HALO = 0（D-35）
assert halo.id_norm.nunique() == len(halo)         # HALO 内部无重复
assert not (set(halo.id_norm) - set(canon.index))  # canon 覆盖全部 HALO
# --- concat：CORE 占 0..30182 原序不动，HALO 从 30183 起 ---
assert len(ladder) == n and ladder.mappedID.tolist() == list(range(n))
assert ladder.iloc[:n_core]["model"].tolist() == core_umi["model"].tolist()
assert ladder["model"].map(normalize).nunique() == n
```

```bash
python -m scale1m.build_ladder --rung 100k --n 100000 \
  --core  stage1BuildTransferGraph/hgraph_ml_v2.pt \
  --halo  $C/selected/selected_halo.parquet \
  --canon $C/canon/hf_canon.parquet \
  --out   $WORK/model_lake/data1m/ladder
```

产物是 `ladder/100k_model_ids.csv`（`mappedID, model, layer, size_b, family, lineage_base`）
和 `LADDER_REPORT.json`（sha256 与全部审计统计）。

梯子的 CORE 侧（0..30182）`size_b`/`family`/`lineage_base` 留空，`layer="core"`（D-38）。
约束 1 要求 CORE 的这些量从冻结图原样复制，梯子不去碰，T4 直接从 `hgraph_ml_v2.pt` 读；
梯子在 CORE 侧只负责钉住 id 与行序。

### 6.3 T3 出闸门与实测统计

- [x] `len(ladder) == 100000` 且 `mappedID` 是 `0..99999` 的连续整数
- [x] `ladder[:30183]["model"]` 与 CORE 的 `mappedID` 序逐项相等
- [x] HALO 与 CORE 交集为 0，湖内 distinct 为 100,000（选样阶段已 `--exclude-core`，D-35）
- [x] `layer` 分布已统计并记录（labeled 6,619 / lineage 25,268 / plain 37,930 / dropped 0）
- [x] `size_b` 缺失率 42.25%，`family` 统计已记录

T2 已经把这一节的量算出来，T3 的任务是对账：

| 量 | T3 实测（v2 HALO） | CORE 参照 |
|---|---|---|
| HALO 来源 | 537,025 候选，配额选出 69,817 | — |
| `size_b` 已知（即带 size 子句） | 57.75% | 47.91%，HALO 高 9.83 pp |
| `size_b` 缺失率 | 42.25% | 52.09% |
| `family` 不同取值 | 9,292 | 207（CORE 用到的） |
| `family == "other"`（规则层） | 0.23% | — |
| `family_id == 0`（过 `FAMILY_MIN_COUNT=3` 折叠后） | 12.78% | 34.16% |
| 落进 CORE 用过的家族 | 48.47% | — |

`family_id == 0` 的比例远低于下面的警戒线，也低于 CORE 自己的 34.16%，`e_fam` 不会塌成一行。
另一半（51.53%）会新建 vocab 行，因此 T4 的 `family_vocab` 从 341 行扩到约 2,000 行。
这不违反约束 1，CORE 已有行的 id 不移位，T4 有断言检查，但 checkpoint 与 vocab 的绑定要同步更新。

如果 `family=Other` 的占比超过 80%，说明 family 推断规则对 HF 长尾失效，HALO 的 `e_fam` 会塌到一行。
这不阻塞实验，Other 本就是设计内的降级路径，但要在报告里写明，
并把它列为 C 轴 frozen 层分析的混杂因素。参照系是 CORE 自己的 34.16%，
「HALO 比 CORE 高多少」比「HALO 是否超过 80%」更有信息量，两者都记。

D-26 的论证在 v2 口径下需要改写，结论不变。v2 的质量门偏好有结构化元数据的模型，
`safetensors` 覆盖率反超 CORE 9.83 pp，偏离方向与 v1 相反。
原来的理由「向 CORE 覆盖率靠拢」不再成立，新的理由是「两个方向都偏离，选偏离更小的那个」，
加名字正则会推到约 80%，偏离更远。

CORE 与 HALO 的 popularity 差异在 v2 后已大幅缓解。v1 时 HALO 是纯头部群体，
v2 的 popularity 配额（head 32.1% / mid 23.2% / long-tail 11.1% / recent 33.6%）把它摊开了。
`downloads` 不是节点特征，所以这不直接影响约束 2，但仍留在 G-B4 的归因清单里。

---

## §7 T4：特征构建（已完成，2026-08-10，watGPU）

执行记录见 [`T4.md`](T4.md) 与 [`T4GPU.md`](T4GPU.md)。

### 7.1 `scale1m/embed_lake.py`

分两段，区别对待：

```python
# ── 段 1：CORE（30,183 行）—— 原样复制，不重算（约束 1）──────────
core = torch.load(CORE_GRAPH, weights_only=False)
x_core        = core["data"]["model"].x                     # [30183, 448]
size_id_core  = core["data"]["model"].size_bucket_id
fam_id_core   = core["data"]["model"].family_id

# ── 段 2：HALO（69,817 行）—— 新算，但用 CORE 的描述符函数（约束 2）──
from scale.modellens_build_graph import model_descriptor
texts   = [model_descriptor(mid, fam, sz) for mid, fam, sz in halo_rows]
e_name  = build_name_embeddings(halo_ids, token_dim=64, seed=42)      # 与 CORE 同 seed
e_desc  = minilm_batched(texts, batch_size=256)                       # all-MiniLM-L6-v2, fp32
x_halo  = np.concatenate([e_name, e_desc], axis=1)                    # [69817, 448]

x = torch.cat([x_core, torch.from_numpy(x_halo)], dim=0)              # [100000, 448]
```

`e_desc` 实际用 batch 256、fp32，与 CORE 走的 `_minilm` 一致；整段只需约 52 秒，没有换配置的理由。
`--batch-size` 留了参数。

size 与 family 的 id：HALO 的 `size_bucket_id` 走 `param_count_to_size_bucket(size_b * 1e9)`，
与 CORE 使用同一套固定常数。`family_id` 走 `load_or_update_family_vocab`，该函数是 append-only：
加载 CORE 的 `family_vocab.csv`，HALO 的新家族只在计数达到 `FAMILY_MIN_COUNT` 时追加新行，
CORE 已有行的 id 不移位。

```python
vocab_before = dict(core["xm0_meta"]["family_vocab"])
vocab_after  = load_or_update_family_vocab(all_families, vocab_path=FAMILY_VOCAB)
for k, v in vocab_before.items():
    assert vocab_after[k] == v, f"family row moved: {k} {v} -> {vocab_after[k]}"
```

### 7.2 执行

```bash
source ~/.mlf_env
cd "$PROJECT_ROOT"
sbatch --export=ALL,RUN_ID=T4_100k_$(date -u +%Y%m%dT%H%M%SZ) scripts/watgpu/embed_lake.sbatch
```

不用 `sbatch --wrap`，因为 `$HOME` 挂了 noexec，需要独立脚本把环境 staging 到 `/tmp`。
实测作业墙钟约 1 分 40 秒，其中 env staging 46.5 秒、`embed_lake` 约 52 秒。
正式产物是 watGPU 上的 `x_m.npy`（sha256 `d7573a02…`），本地那份留作对照。

### 7.3 T4 出闸门与实测

- [x] `torch.equal(x[:30183], x_core)`，CORE 特征逐字节未变
- [x] `x.shape == (100000, 448)`，无 NaN、无 Inf；HALO 全零行 0 个，L2 norm 均值 1.076，范围 [1.013, 1.285]
- [x] `family_vocab` 只增不移
- [x] 行序 checksum：随机抽 100 个 HALO 行，按 `ladder` 里的 id 重算 `e_name`，与 `x[i,:64]` 比对相等
- [ ] G-B4：CORE/HALO 可分性 AUC ∈ [0.5, 0.75]，实测 0.9729，未通过原门，按 D-39 改判

G-B4 的诊断是这一层唯一能发现描述符出问题的检查，因此写进主流程默认执行，不是可选旁路脚本。
做法是用逻辑回归拿 `x` 分类「这一行是 CORE 还是 HALO」，30,183 对 30,183 平衡采样，50/50 train/test：

| 用哪部分特征 | 测试集 AUC |
|---|---|
| 完整 `x`（448 维） | 0.9729 |
| 只用 `e_name`（前 64 维，哈希平均） | 0.7093 |
| 只用 `e_desc`（后 384 维，MiniLM） | 0.9714 |

为了判断该不该回滚描述符，又对同一批抽样行（每组 6,000）重新生成简化描述符再嵌入，做子句消融：

| 描述符内容 | AUC | 相对只有名字的增量 |
|---|---|---|
| 只有名字 | 0.8472 | — |
| 名字 + size 子句 | 0.9313 | +0.084 |
| 名字 + family 子句 | 0.9146 | +0.067 |
| 完整（现状） | 0.9655 | +0.118 |

三条读法：地板是 0.847 而不是 0.5，把描述符退到只剩清洗过的模型名字，两群在 MiniLM 空间里仍可线性分开，
这部分来自 CORE 是 ModelLens 的基准语料、HALO 是 2026 年按配额爬的 HF 均衡样本，
命名习惯、组织构成与年代都不同，回滚描述符改不动它；流水线自己贡献 0.118，
两条子句的单独增量之和大于联合增量，说明它们捕捉的信息重叠；
计划原先把 size 覆盖列为唯一能调的旋钮，实测 family 子句的贡献是同一量级（0.067 对 0.084）。

约束 2 要防的是流水线人为制造两群之间的差异，从消融看这一层是干净的：描述符函数确实是同一个。
原门写成绝对 AUC ≤ 0.75，预设了两群本来接近同分布，而实际不是。
按 D-39，G-B4 改为报两个数：绝对可分性 0.9729，流水线增量 0.9655 − 0.8472 = 0.118，总体基线 0.8472。

AUC 的归因通道按实测量级排序：

1. family 子句覆盖率差 34 pp（D-40）。CORE 的 intake 里 33.98% 的模型没有 family 字符串，
   `model_descriptor` 遇到空 family 会整段省略 `family <x>`，因此 CORE 只有 66.02% 的描述符带 family 子句，HALO 是 100%。
2. size 子句覆盖率差 +9.83 pp（D-26），这是设计内的选择，且符号与 v1 相反（v1 时 HALO 更少，v2 时更多）。
3. CORE 有 19.35% 的裸名字（`bert-base-uncased`、`#-shots` 这类没有组织前缀的），HALO 是 0%，
   因为 HF 上所有仓库都是 `org/name` 形式。
4. 名字长度：CORE 平均 38.9 字符，HALO 33.8，HALO 短 5.1 字符。
   整段描述符方向相反，HALO 长 5.0 字符，因为两条子句把名字短掉的部分补回来还有余。
   报告里引用「HALO 名字更短」时要说清是哪一半，两个数不矛盾但容易被读成矛盾。
5. popularity 差异，v2 后已大幅缓解。

这个结果对下游有两个约束，都要写进报告。第一，不能说「扛住了 83 倍同分布干扰」，
HALO 不是 CORE 的同分布扩充，它是另一批模型；能说的是「候选池从 30,183 扩到 100,000 之后 gold@10 下降了多少」，
这是关于检索在更大更杂的湖里表现的陈述。第二，冷启动分析要把它列为混杂因素：
`frozen` 层占 HALO 的 53.48%，而 CORE 与 HALO 本身可分，观察到的分层差异里有多少来自「冷」、
有多少来自「是另一群模型」，从现在的设计里分不开。这两条在看到 T8 数字之前先写下来，
是为了避免顺着数字挑口径。

---

## §8 T5：建图

### 8.1 `scale1m/build_graph_rung.py`（改自 `scale/modellens_build_graph.py`）

| 边类型 | 100K 上怎么建 | 边数 |
|---|---|---|
| `trained_on` + rev | 从 CORE 原样复制（HALO 无监督边） | 312,986，与 CORE 相同 |
| `similar_to` (d–d) | 从 CORE 原样复制（数据集节点完全不动） | 约 192,060 |
| `is_base_of` + rev | HALO 的 `lineage_base` 对 100K 的 id 表做 hashmap join | 实测 16,385 |

```python
# 血缘边：规范化 id 的 hashmap join，O(N)
id2idx = {normalize(m): i for i, m in enumerate(ladder["model"])}
src, dst = [], []
for i, base in zip(ladder.index, ladder["lineage_base"]):
    j = id2idx.get(normalize(base))
    if j is not None and j != i:
        src.append(j); dst.append(i)          # base --is_base_of--> derivative
```

必须产出 `lineage_stats.json`，无论结论好坏（`plan.md §3.2` 的要求）：

```json
{"total_edges":?, "core_core":?, "core_halo":?, "halo_halo":?,
 "n_models_with_lineage":?, "n_components":?, "largest_component":?,
 "halo_lineage_coverage": ?}
```

这是 D-8 的第一次真实检验。HALO 里声明 `base_model` 的有 27,672 条（39.64%），
其中能在 100K 湖内解析的是 16,385 条，构成 `is_base_of` 边：`core_halo` 606 条、`halo_halo` 15,779 条、
`core_core` 0 条。相对 P1 时期名字精确匹配得到的 42 条，是约 390 倍。
按 v1 前缀口径估算的 26,297 条比这个数高 38%，差异来自近重复上限削掉了血缘最密的一群
（同一个 base 的 20 个 GGUF 变体各连一条）。这是取舍不是退步，那些边对 hub 过平滑是负担而不是信息，
但两个数在报告里要并列出现。

`relation` 从 `baseModels` 直接取得（D-37），quantized/adapter/finetune/merge 是 HF 的结构化字段，
`r_mm'` 的离散有序权重不必再从名字推断。重建后的实测分布是
finetune 10,226 / quantized 4,360 / adapter 1,608 / merge 76 / unknown 115，合计 16,385。

这个结果带一个必须一起说明的限制：`core_core = 0`、`core_halo` 只有 606，
血缘结构几乎全部长在 HALO 内部，而 gold 标签只存在于 CORE。
因此 §11.3 C 轴的第三问（`cold` 是否显著优于 `frozen`）不能用 `gold@10` 回答，只能用表示质量类指标回答。

`core_core = 0` 的成因与一个明确的不作为：CORE 的 30,183 行来自冻结的 `hgraph_ml_v2.pt`，
本来就没有 HF `cardData`。我们爬到了 1,199 个 CORE 模型的 HF 记录（含其 `base_model`），
但不用它给 CORE 子图加边（D-28），因为加了就等于改动 CORE，R0/R1/R2 之间「N 是唯一变量」不再成立。

### 8.2 执行

```bash
python -m scale1m.build_graph_rung --rung 100k \
  --core   $WORK/model_lake/data1m/graphs/hgraph_ml_v2.pt \
  --ladder $WORK/model_lake/data1m/ladder/100k_model_ids.csv \
  --feats  $WORK/model_lake/data1m/feats/100k \
  --canon  $C/canon/hf_canon.parquet \
  --out    $WORK/model_lake/data1m/graphs/hgraph_100k.pt
```

文件约 250 MB，构建 3–10 分钟。
`--canon` 不能漏：第一次建图时 `hf_canon.parquet` 没有传到远端，血缘 `relation_id` 全部落成 `unknown`，
D-37 拿回来的 94.9% 血缘声明没有进图。按 D-44 已重建（`7fbc3c47…` → `16f35214…`），
逐张量比对确认五种边的 `edge_index`/`edge_attr` 与 `model.x`/`size_bucket_id`/`family_id`/`dataset.x` 全部相同，
唯一不同的是 `relation_id`。

### 8.3 T5 出闸门（对应 §12 的 G-B 系列）

```python
# scale1m/verify_rung_graph.py --rung 100k  —— 一条命令跑完全部
assert data["model"].num_nodes == 100_000
assert data["dataset"].num_nodes == 9_603
assert data["model"].x.shape == (100_000, 448)
assert data["dataset"].x.shape == (9_603, 458)
assert len(data.edge_types) == 5
assert not torch.isnan(data["model"].x).any()
# 约束 1 的六条断言（见 §1）
# xm0_meta / xd0_meta 与 CORE 逐键一致（family_vocab 只允许变长）
```

- [x] `verify_rung_graph --rung 100k` 全绿（33 条门禁已在实际训练使用的那张图上核对）
- [x] `stage1BuildTransferGraph/check_stage2_contract.py` 通过
- [x] `lineage_stats.json` 已产出
- [x] 划分不变性：用 `make_root_aware_splits(data, roots, split_seed=0)` 在 100K 图与 CORE 图上各跑一次，
      test 边集合的 sha256 相同（实测 `66c81823cb731063…`，65,968 边）。
      HALO 没有监督边，划分只由 CORE 决定，因此这两个 sha256 相同是预期结果，不同则说明有 bug。

---

## §9 T6：训练

### 9.1 `scale1m/train_rung.py`（改自 `scale/export_ours.py`）

复用 `export_ours` 的全部逻辑，增加 `--expect-n`、`--chunked-infer`、`--sparse-M`、`--loader fanout`、
`--amp`、`--resume`、`--out`。

配置沿用 `l1l3b_config()`（[export_ours.py:50](../../scale/export_ours.py#L50)）不做改动：
L1 全湖 logQ 采样 softmax、L3 native task、`global_n_neg=256`、`batch_size=1024`。

`--amp` 在 R2 不启用（D-42），推到 R3/R4 显存确实不够时再加，且加的时候单独跑一次锚点对拍。
理由是 T6 第 1 步的 12K 锚点要证明「换了环境、开了 T0 的四个开关，模型还是原来那个」，
同一批改动里既换环境又换数值精度，锚点一偏就无法归因；
而实测 12K 峰值 0.583 GB、100K 峰值 1.737 GB，显存目前不是约束。

### 9.2 三步走

```bash
# ── 步 1：R0 锚点（12K）在集群上复现 ──────────────────────
RUNG=12k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
#    ⇒ 判据见 G-C2。不过则停下查环境、AMP 与 loader，不改配置去凑

# ── 步 2：R1（30K 全 CORE，D-9 解除）───────────────────────
RUNG=30k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
#    ⇒ 这是第一次跑通全 CORE，本身是独立交付

# ── 步 3：R2（100K）先冒烟再全量 ────────────────────────────
RUNG=100k SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh    # 冒烟
#    ⇒ loss 下降、无 NaN、显存与墙钟记录，据此定 --time
RUNG=100k SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh    # 全量，seed 0/1/2 各一次
```

监督边数三档不变（312,986），代价增量来自采样与推理，不是来自监督量，这是监督冻结设计的一个附带好处。
实测单次 25 epoch 的墙钟：12K 约 92 秒、30K 约 431 秒、100K 约 109 秒（三 seed 平均，见 T7.5 §5.1）。

`--seed` 的语义是 `split_seed`，`init_seed` 恒为 0（沿用 `export_ours.py`，不改，D-43）。
因此多 seed 给出的是划分噪声区间，不是初始化噪声区间，报数字时要写明是哪一种。
实测三个 100K seed 的 held-out 数据集数是 734 / 775 / 936，`tau_macro` 为 0.180 / 0.077 / 0.217，
三个 run 的测试集并不相同。

### 9.3 checkpoint

每 5 epoch 及验证边界保存 checkpoint。内容包括 model、optimizer、scheduler、AMP gradient scaler、epoch、global step、best metric 与 early-stopping 状态、Python/NumPy/CPU torch/CUDA torch 随机状态、resolved config 与数据版本；可行时同时保存 sampler/DataLoader 进度。模型状态必须包含 encoder 的两张 embedding 表，并记录 `family_vocab.csv` 路径与 SHA-256、size bucket 常数版本、`e_name` seed、`token_dim` 与 encoder 名（CLAUDE.md Step 6 的要求）。

checkpoint 先写临时文件，完成 flush/close 后原子 rename 为 `checkpoint_step_<N>.pt`；`last.pt`/`latest` 仅在目标文件成功后更新。运行目录保留 `best`、`latest` 与至少一个历史 checkpoint，不覆盖唯一有效副本。

resume 解析顺序固定：显式 `--resume <path>`；`$OUT/ckpt/last.pt` 或 `latest`；新运行。
日志记录最终选择的 checkpoint。恢复前校验模型结构、resolved config、数据与划分标识、`family_vocab` 与 size bucket 版本，不兼容时明确失败。Slurm preemption/requeue 继续使用同一 logical `RUN_ID`，恢复 optimizer/scheduler/scaler 与 step 后再推进，避免重复 metrics。完整训练提交前需通过 save、reload、resume 后 step 连续递增的测试。

### 9.4 T6 出闸门

- [x] 机制门：loss 单调下降、无 NaN、两张 embedding 表梯度非零、frozen `x` 梯度为 `None`
      （2026-08-11 六个 run 加 2026-08-14 六个新 run，12/12 通过）
- [ ] 锚点门 G-C2：见 §12，实测不通过，处理方案待定
- [x] R1 训练成功，D-9 正式解除（30K，25 epoch，loss 18.9898 → 15.0575）
- [x] `MANIFEST.json` 已写，含 `peak_gpu_mem_gb` 与 `wallclock_s`，本地 evidence 合约复验通过
- [x] 收尾核对（2026-08-14，[`T6more.md`](T6more.md)）：T5 门禁 33/33 已在实际训练使用的图上核对；
      as-run 代码对账 105/105 逐字节相同；12k/30k 的 `family_vocab` 绑定补齐（`d4e6b682…`）；
      `git_head` 修正为 `304e11b8`，此前六个 run 里是空串，根因是 git 跑在仓库根的上一级
- [x] 100K 图按 D-44 重建后，四个 100K run 已在新图上重跑

---

## §10 T7：导出与索引

```bash
python -m scale1m.export_rung --rung 100k --run $OUT \
  --chunk 50000 --hnsw-M 32 --ef-construction 200 --hnsw-threads 8
```

| 文件 | 内容 | 检查 |
|---|---|---|
| `z_m.npy` | `[100000,128]` 全图前向，服务用 | 行序断言 |
| `z_m_eval.npy` / `z_d_eval.npy` | held-out 前向，A 轴数字只能用这个 | 泄漏门 |
| `hnsw_100k.bin` | HNSW 索引 | recall ≥ 0.99 |
| `model_ids.csv` | `mappedID → unique_model_id` 快照 | 与 ladder 逐行相等 |
| `gold_cands.npz` | held-out gold 标签 | 与 CORE 相同 |
| `ef_tuning.json` | iso-recall 二分过程 | — |

泄漏门（G-D2）：P4 遇到过全图 `z_d` 让 held-out 查询看到自己的监督边的情况，
`gold@10` 从 0.42 虚高到 0.61。

```python
assert gold10(z_eval) < gold10(z_full), "A 轴必须用 held-out 前向的 z_*_eval"
```

T7 的七个 run 全部通过，两者间距 0.056–0.169（12K 0.387 对 0.526、30K 0.203 对 0.372、100K 0.207 对 0.313）。

行序门（G-D3）：随机抽 100 个 `mappedID`，反查 `model_ids.csv` 的 `unique_model_id`，与 `ladder` 逐项相等。
100K 上已真查并通过（抽 100 加全表复核）；12k/30k 没有 ladder，记为 skip 而不是 pass。

---

## §11 T8：三轴评测

### 11.1 A 轴：精度

```bash
python -m scale1m.eval_rung --rung 100k --run $OUT --expect-n 100000
```

单 seed 主表（`split_seed=0`）：

| N | gold@1 | gold@10 | top3@10 | gold-gap@10 | root_gold@10 | median gold rank | median rank / N | vs random |
|---|---|---|---|---|---|---|---|---|
| 12,000（P3 存档，改造前） | 0.0948 | 0.4159 | 0.5319 | 0.5087 | 0.3920 | 21 | 1.75×10⁻³ | 4.99×10² |
| 12,000（T7 实测，T0 开关全开） | 0.0870 | 0.3868 | 0.5261 | 0.5532 | 0.3493 | 23 | 1.917×10⁻³ | 4.64×10² |
| 30,183 | 0.0586 | 0.2030 | 0.2766 | 0.3256 | 0.1929 | 96 | 3.197×10⁻³ | 6.13×10² |
| 100,000 | 0.0436 | 0.2071 | 0.3229 | 0.3065 | 0.1923 | 98 | 9.800×10⁻⁴ | 2.07×10³ |

三 seed 均值（T7.5 §5.1，`approx` 臂即出货配置）：12K 0.3633 ± 0.0206、30K 0.2295 ± 0.0252、
100K 0.2145 ± 0.0467。seed 之间的散布大于多数被比较的差值，因此 A 轴的任何数字都要带 seed 区间报告，
单 seed 的两两比较不作为结论。

第一行与第二行是同一个 `split_seed=0`、同一批 517 个查询，差 −0.0291，不可能是划分噪声，G-C2 因此不通过。
T7.5 已排除对比损失采样与扇出上限两个近似作为原因，差异归到环境与采样后端
（P3 没有 `pyg-lib`，走纯 Python 全邻域；现在走 PyG 的 C++ 采样器）。
在口径确定之前，第二至四行不作为最终数字对外。

`vs random` 的定义是 `gold@10 / (10/N)`，100K 时随机基线是 1×10⁻⁴。
`median rank / N` 是核心列而不是 `gold@10`，因为它衡量 gold 在池子里的相对位置，是唯一跨 N 可比的量。

预注册预测（在跑之前写下，用于防止事后编故事）：若表示的判别力与规模无关，ranks 随 N 按比例放大，
则 100K 的 `gold@10` 约等于 12K 的 `gold@1.2`，即 0.10–0.12；
若表示对 HALO 这类无标签长尾有额外判别力（它们多是离题模型，容易被推远），则会高于这个带，落在 0.15–0.30。
落在 0.10–0.30 视为符合预期；超过 0.35（几乎没掉）应先查三处再下结论：候选池是否真的是 100,000（约束 3）、
HALO 的描述符是否用对（约束 2 与 §7.3 的 AUC 诊断）、泄漏门 G-D2；
低于 0.05 则是表示在放大后失效，照实报告并做 C 轴的分层归因。
实测 0.2071（三 seed 0.1720–0.2645）落在预注册区间内。

`displacement_quality` 是 D-10 的副产品：

```
对每个 held-out 查询，取排在 gold 前面的 HALO 模型；
其中 layer == "labeled" 的那些，比较它们在该数据集上的实测精度与 gold 的精度：
  - 显著更低 ⇒ 真错误
  - 接近或更高 ⇒ 挤占其实发现了 CORE 语料没标注的好模型，gold@10 低估了真实效用
两种情形都照实报，并写清占比。
```

v2 口径下 `layer=labeled` 的 HALO 有 6,619 条（9.48%），样本基数是 v1 的 4 倍，
这个分析可以给定量结论，但仍要连分母一起报（「N 个挤占者中 M 个 labeled」）。

### 11.2 B 轴：iso-recall 延迟

测量协议比数字本身更重要：

1. 先用 `tune_ef_for_recall(target=0.99)` 二分出最小 `ef_search`，然后才计时。
   否则「你只是把 recall 调低换速度」这一条反驳无法回应。
2. 单线程测量：`idx.set_num_threads(1)`，逐查询、禁用 batch。
3. warmup 100 次丢弃，再测 1000 次。
4. 报 p50 / p95 / p99，不只报均值。
5. 同一次作业内跑完 R0/R1/R2 三档，跨节点的 CPU 差异会污染曲线。
6. 另报多线程 QPS，部署关心的是这个量。

| N | ef_search | recall@50 | p50 (ms) | p95 | p99 | 我方 O(N) 全扫 p50 | 加速比 | 索引 (MB) | ModelLens Θ(N) 外推 p50 | vs 外推 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12,000 | 52 | 0.9906 | 0.0149 | 0.0212 | 0.0276 | 0.2404 | 16.2× | 8.8 | 16.60 | 1,117× |
| 30,183 | 52 | 0.9901 | 0.0162 | 0.0221 | 0.0255 | 0.5875 | 36.2× | 22.1 | 40.80 | 2,516× |
| 100,000 | 50 | 0.9921 | 0.0160 | 0.0226 | 0.0264 | 1.8856 | 118.0× | 73.2 | 133.72 | 8,371× |

实测（作业 `1511164`，三档同一进程）：HNSW 的 p50 对 N 基本持平，幂律指数 α = 0.032；
全扫对 N 线性（N 涨 8.33 倍，全扫涨 7.84 倍），加速比随 N 单调放大，16× → 36× → 118×。
三档的 `ef_search` 都停在下界附近（52/52/50），说明 100K 还没有让 ANN 付出代价，
曲线斜率要等 R3/R4。详见 [`T8.md`](T8.md) §1。

ModelLens 那一列标注为外推而非实测（D-16）。外推方法要写清：
用 P4/P5 在 1K–12K 实测的点拟合 Θ(N) 直线并给出区间。

曲线拟合先算但不下结论：对 p50 分别拟合 `a·log N + b` 与 `a·N^α + b`，报 α 与置信区间，
三个点太少，裁决等 R3/R4。

### 11.3 C 轴：冷启动分层

| 层 | 定义 | 100K 预测 | 100K 实测 | 层内平均余弦 | 有效维度 (/128) | 质心 cos(warm) |
|---|---|---:|---:|---:|---:|---:|
| warm | `trained_on` 度 ≥ 10 | 5,765（5.76%） | 5,765（5.76%） | 0.553 | 3.16 | — |
| cool | 度 1–9 | 24,369（24.37%） | 24,369（24.37%） | 0.626 | 4.49 | 0.666 |
| cold | 度 = 0 且有血缘 | 16,385（16.39%） | 19,852（19.85%） | 0.935 | 4.12 | 0.863 |
| frozen | 度 = 0 且无血缘 | 53,481（53.48%） | 50,014（50.01%） | 0.815 | 4.34 | 0.796 |

`warm` 与 `cool` 完全来自 CORE，是确定值，实测逐个相符，这同时交叉验证了行序与分层口径。
`cold`/`frozen` 的预测偏了，原因可以指认：预测把「血缘边数 16,385」当成了节点数，
实际有血缘且度为 0 的模型是 19,852，等于 `n_models_with_lineage` 20,135 减去 283 个本身有监督边的 CORE 模型。
结论方向不变，`frozen` 仍占一半。

C 轴要回答三个问题，实测结果如下：

1. `frozen` 层能被检索到吗？它们的 `z_m` 完全来自 `[e_name‖e_desc‖e_size‖e_fam]` 加空邻域，
   正是 edge-dropout 训练的降级路径。实测能被检索到，它占了挤占 gold 的模型的 44.8%，
   问题不是进不去，而是进得太容易。
2. `frozen` 层是否塌缩？测层内平均余弦、有效维度与到 warm 层的分布距离。实测有塌缩。
3. `cold`（有血缘无标签）是否显著优于 `frozen`？这一问必须换指标：
   血缘几乎全在 HALO 内部（`core_core=0`，`core_halo` 仅 606），而 gold 标签只存在于 CORE，
   `cold` 节点里几乎没有 gold 候选，用 `gold@10` 回答会得到「两边都约等于 0」的假平局。
   改用表示质量回答：层内平均余弦、有效维度、与 warm 层的分布距离，
   以及「同一个 base 的派生模型之间的余弦对随机对的余弦」，即血缘边是否真的把派生族拉近，同时没有拉塌。
   实测的答案与预期相反：`cold` 的层内平均余弦 0.935 高于 `frozen` 的 0.815，即 `cold` 塌得更厉害，
   与 CLAUDE.md 风险三「血缘边把派生族塌成一个点」一致。
   血缘的净分离随 N 收窄（+0.31 → +0.35 → +0.15），而且是被随机对余弦从 0.61 涨到 0.74 吃掉的，
   整个空间在随 N 收紧。详见 [`T8.md`](T8.md) §2。

这个结果记为 G-E1。它指向的补救方向（加强同 hub 负样本、减一层、调 edge-dropout）属于新实验，
不在 R2 范围内，但要写进 `R2_EXECUTION.md`。

另外，§7.3 的可分性结果是这一节的混杂因素：CORE 与 HALO 本身可分，
分层差异里有多少来自「冷」、有多少来自「是另一群模型」，现在的设计分不开。

### 11.4 D 轴：工程

| 指标 | 12K | 30K | 100K |
|---|---|---|---|
| 端到端建库墙钟（爬取→嵌入→建图→训练→索引） | | | |
| 峰值 GPU / CPU 内存 | | | |
| 索引磁盘 | | | |
| 增量上线一个新模型（特征→前向→`add_items`） | | | |
| 多线程 QPS | | | |

最后一行是产品级论据：新模型上线是 O(1)，不重训、不重建索引，
而 ModelLens 的 id-embedding 架构做不到，因为新模型没有 id 行。

---

## §12 出闸门总表

| ID | 阶段 | 判据与当前状态 | 不过怎么办 |
|---|---|---|---|
| G-W1 | T1 | SSH access、quota、大文件目录、partition/account 均有实测记录 | 暂停上传与提交，向管理员确认 |
| G-W2 | T1 | 环境依赖通过，CUDA 在 allocation 内可见，tiny batch forward/backward 通过 | 修复环境或资源申请后重跑 smoke test |
| G-W3 | T1 | Git commit、数据 manifest/checksum 已记录，resolved config 无 Windows 路径 | 修正集中配置并重新同步与校验 |
| G-W3' | T1/T6 | 已补齐。`git_head` 曾在全部六个 run 里是空串（根因是 git 跑在仓库根的上一级），已修并实测为 `304e11b8`；as-run 代码对账 105/105 逐字节相同 | 停止并补齐审计 |
| G-W4 | T1/T6 | checkpoint save/reload/resume 后 step 连续，状态完整恢复 | 不提交 25-epoch 作业，先修复恢复逻辑 |
| G-W5 | T1 | tiny `sbatch` 完成，log、metadata、exit code 与 job ID 正确落盘 | 修正 Slurm wrapper 与输出目录 |
| G-W6 | T8 | checkpoint、metrics、分片、commit/config 与最终 Slurm 状态一致 | 运行保持未验收状态并补齐审计 |
| G-A1 | T0 | 未通过。T0 阶段从未执行，第一次跑到是在 T7，见 G-C2 | 见 G-C2 |
| G-A2 | T0 | 已过。分块推理与全图前向 `max\|Δ\| = 0.00e+00`（12K/30K，逐位相同） | 查 `n_id` scatter |
| G-A3 | T0 | 已过。重写后的 `five_metric_eval` 与 P3 存档逐位相同，与 `global_metrics` 的 `gold_rank` 0/517 不一致；T7 的七个 run 复验 PASS | 查流式打分改写 |
| G-A4 | T0 | 已补做。D-14 要求的对拍由 T7.5 以三档 × 三臂 × 三 seed 的形式完成，未跑 n_neg=1024 | — |
| G-B1 | T5 | 已过（远端正式图上实测，2026-08-13）：约束 1 的六条断言全过 | 停止 |
| G-B2 | T5 | 已过：`check_stage2_contract passed` | 停止 |
| G-B3 | T5 | 已过：test 边 sha256 `66c81823cb731063…`，65,968 边，`rung == core` | 停止排查 |
| G-B4 | T4 | 未过原门，已按 D-39 改判。实测绝对 AUC 0.9729，流水线增量 0.118，只用名字的基线 0.8472 | 不回滚描述符；按 §7.3 的通道归因并在报告与 C 轴中列为混杂因素 |
| G-T2a | T2 | 已过。v1 快照门：150,000 条 / 3 片 sha256 复算一致 / 只读 / 快照日 2026-08-03 / 0 重复 | — |
| G-T2b | T2 | 已过。v2 总体门 G1–G20（`verify_raw --selected`），全部按实际选出的 69,817 计算：18/20 硬门过，G14/G15 受联合约束所限并已定量披露 | 停止，未过门的总体不称为 frozen |
| G-T2c | T2/T3 | 已过：`CORE ∩ HALO = 0` 且 `\|CORE ∪ HALO\| = 100,000`。G1–G20 查不到这一条（G1 只查 HALO 内部重复），由 `build_ladder` 断言 | 用 `--exclude-core` 重跑选样（D-35） |
| G-T3 | T3 | 已过：梯子 100,000 行、`mappedID` 连续、CORE 前缀逐项相等、全表 id 去重 100,000；`ladder_sha256` 已记 | 停止，行序错位不会自己暴露 |
| G-C1 | T6 | 已过 12/12（2026-08-11 六个 + 2026-08-14 六个）：机制门（loss 下降、无 NaN、梯度边界） | 回到 T0 |
| G-C2 | T7 | 未通过。同一个 `split_seed=0`、同一批 517 查询，P3 存档 0.4159，本次 0.3868（−7.0%）；三个 split seed 实测 0.3488 / 0.3868 / 0.3544（mean 0.3633，sd 0.0206），0.4159 在区间之外。T7.5 已排除两个近似为原因 | 待裁定：是否按 T7.5 的建议改为「12K 的 3-seed 区间覆盖历史值」，并把差异归到环境与采样后端另立基线。在裁定前不改配置去凑 |
| G-D1 | T7/T8 | 已过：100K 三个 run 的 `z_m.shape[0] == 100000` | 停止，见约束 3 |
| G-D2 | T7/T8 | 已过：七个 run 全过，间距 0.056–0.169 | 停止 |
| G-D3 | T7/T8 | 已过：100K 上真查并通过（抽 100 加全表复核 vs ladder）；12k/30k 无 ladder，记为 skip | 停止 |
| G-D4 | T8 | 已过：完整协议下三档 recall@50 为 0.9906 / 0.9901 / 0.9921，ef 52/52/50。ef 仍停在下界，说明 100K 未逼 ANN 付出代价 | 调 ef 或 M；调不上去查过平滑 |
| G-D5 | T7/T8 | 记为违反：单 seed 下 100K 0.2071 高于 30K 0.2030（同一批 734 查询）。按 T7.5 的三 seed 均值 30K 0.2295 高于 100K 0.2145，方向恢复正常，该记录是否解除需按新口径重新裁定 | 若按新口径仍上升，先查泄漏与采样偏置，不作为正面结果叙述 |
| G-D6 | T7/T8 | 已过：100K 实测 0.2071（三 seed 0.1720–0.2645），落在预注册的 [0.10, 0.30] 内 | 超出上界先查 bug，见 §11.1 |
| G-E1 | T8 | 新增，实测发现：`cold` 层塌缩 0.935 高于 `frozen` 0.815，有效维度 3–5/128，随机对余弦随 N 从 0.61 涨到 0.74 | 属新实验（同 hub 负样本、减层、调 edge-dropout），不在 R2 范围，写进 `R2_EXECUTION.md` |

---

## §13 失败排查

| 症状 | 最可能原因 | 先查哪里 |
|---|---|---|
| 作业长期 `PENDING` | partition/account 无权限，或 GPU/CPU/RAM/constraint 请求过严 | `squeue -t PENDING` 与 `scontrol show job -dd`；按 pending reason 修正请求，避免反复取消重投 |
| 环境或 import 失败 | batch shell 未激活正确环境，或依赖与节点 CUDA 不兼容 | `which python`、`python -m pip check`、环境列表与 Slurm 激活段 |
| 远端出现 `C:\` / `D:\` | local-only path 进入代码或 resolved config | 搜索仓库与配置，改为 `PROJECT_ROOT`/`DATA_ROOT`/`OUTPUT_ROOT` |
| 作业被 preempt 或 requeue | checkpoint 或 logical `RUN_ID` 恢复不完整 | 检查 resume 日志、step 连续性、optimizer/scheduler/scaler 与 metrics 去重 |
| `gold@10` 几乎没掉（>0.35） | 候选池仍是 30K，或 HALO 描述符用错，或泄漏 | G-D1 → G-B4 → G-D2 |
| `gold@10` 塌到约 0 | HALO 特征 NaN 或全零；或 `x[:30183]` 被覆盖 | G-B1；查 `x[30183:]` 的 norm 分布 |
| CUDA OOM（训练） | 扇出上限没生效，或对比损失仍是全 N² | 先记录失败配置与峰值；修正采样，或减小 per-device batch 并用 gradient accumulation 保持 effective batch，学习率变化要显式登记 |
| CUDA OOM（导出） | 分块推理没走上 | 确认 `--chunked-infer` 生效 |
| CPU OOM | 稀疏 `M` 没生效，或 `scores_by_q` 仍被物化 | 查 `M.is_sparse` 与流式打分；依据 `MaxRSS` 调整 worker/prefetch 或 `--mem` |
| HNSW recall 上不去 | hub 附近过平滑（CLAUDE.md 风险之三） | 先调高 `ef_search`；仍不行则加强同 hub 负样本或减一层。100K 上的第一嫌疑是量化搬运仓库，它们共享 `e_name` 的组织 token，又通过 `base_model` 指回各自原模型 |
| 血缘边数远少于 16,385 | `base_model` 规范化不匹配（大小写或前缀），或 join 写成了对 CORE-only 的表 | 与 `T3.md` §4.2 的三个数（16,385 / core_halo 606 / halo_halo 15,779）逐项对账；抽 20 条人工比对 `normalize()` |
| 血缘 `relation_id` 全是 `unknown` | 建图时没有传 `--canon`，`hf_canon.parquet` 不在远端 | 查 `candidates_v2/canon/` 是否存在；见 D-44 |
| `size_b` 100% 缺失 | 爬取走了 `full=true` 而不是 `expand[]` | 查 `PROVENANCE.json` 的 `expand` 字段里有没有 `safetensors`；有则查 T3 的取值路径 |
| 划分 sha256 不同 | CORE 的 `trained_on` 边序被改了 | 查 G-B1 的 `edge_index` 断言 |
| 作业撞 walltime | 单 epoch 估错或未按期 checkpoint | 从有效 checkpoint 续跑；按已完成 step 实测申请时限，或拆为可恢复阶段 |
| `family_vocab` 断言失败 | `load_or_update_family_vocab` 不是 append-only 调用 | 确认传了 CORE 的 `vocab_path` |

---

## §14 产物目录

```
$WORK/model_lake/          （本地为 %MLF_DATA_DIR% = D:\research\model_lake\data）
├── data1m/
│   ├── raw/           v1 冻结件：hf_models_{00000..00002}.jsonl.gz（150,000 条 / 24.8 MB / 只读）
│   │                     PROVENANCE.json（只读）  SHARDS.json  CURSOR.json
│   │                     —— 审计件与候选源，不再是 HALO 的定义（D-30）
│   ├── v1_audit/      annotated_v1.parquet（对照用）
│   ├── candidates_v2/ hf_models_000{00..11}.jsonl.gz（537,025 条 / 64.7 MB / 只读）
│   │                     membership/*.tsv.gz（532 个查询各自的命中列表，溯源合并依据）
│   │                     PROVENANCE.json  PLAN_STATE.json  SHARDS.json
│   │                     annotated.parquet  ANNOTATE_REPORT.json  AUTHOR_STATS.json
│   │                     selected/       selected_halo.parquet/.csv（T3 的 HALO 输入）
│   │                                     SELECTION_MANIFEST.json  DEFICITS.json  EXIT_GATES.json
│   │                     selected_rerun/ G19 的确定性对照
│   │                     canon/          hf_canon.parquet（537,025 行）CANON_REPORT.json
│   │                                     —— 建图需要，须同步到远端（D-44）
│   ├── ladder/        100k_model_ids.csv（100,000 行 / 6.9 MB，sha256 ccf288ae…）
│   │                     LADDER_REPORT.json
│   ├── feats/100k/    x_m.npy（sha256 d7573a02…）  size_bucket_id.npy  family_id.npy
│   │                     family_vocab.csv  FEATS_REPORT.json  halo_descriptors_head.txt
│   └── graphs/        hgraph_ml_v2.pt(CORE)  hgraph_ml_v2_sub.pt(R0)
│                      hgraph_100k.pt（16f35214…）  hgraph_100k_norel_7fbc3c47.pt（归档）
└── runs/R2_s{0,1,2}_<date>/
    ├── MANIFEST.json  slurm-*.out
    ├── ckpt/          last.pt  best.pt  family_vocab.csv
    ├── exports/       z_m.npy  z_m_eval.npy  z_d_eval.npy  hnsw_100k.bin
    │                  model_ids.csv  gold_cands.npz  ef_tuning.json
    ├── metrics/       a_axis.json  b_axis.json  c_axis.json  d_axis.json
    │                  lineage_stats.json  displacement_quality.json
    ├── stdout/        train.log  evaluate.log
    └── metadata/      run.txt  resolved_config.*  nvidia-smi.txt
                       pip-freeze.txt  uncommitted.patch（仅工作树非 clean 时）

codes/ModelLakeFishing/
├── scale1m/           __init__.py  hf_crawl.py  query_plan.py  taxonomy.py
│                      annotate_candidates.py  select_balanced_halo.py
│                      gates.py  report_distribution.py  verify_raw.py
│                      hf_canonicalize.py  build_ladder.py
│                      embed_lake.py  build_graph_rung.py  verify_rung_graph.py
│                      train_rung.py  export_rung.py  eval_rung.py  write_manifest.py
│   └── tests/         test_hf_crawl.py / test_taxonomy.py / test_selection.py 等（99 passed）
├── configs/           halo_quotas.json（全部配额与上限的唯一出处，`halo-quota-1.2`）
│                      watgpu.example.env（仅占位符）
├── scripts/watgpu/    bootstrap_env.sh  smoke_test.sbatch  embed_lake.sbatch  train_rung.sh
│                      evaluate_rung.sh  build_index.sh  crawl.sh
│                      collect_run_metadata.sh  download_results.ps1
└── docs/1M/           T0.md  T0_runs/  T2.md  T2_runs/  T3.md  T4.md  T4GPU.md
    │                  T5.md  T6GPU.md  T6more.md  T7.md  T7.5.md  T8.md
    └── S4/            R2_EXECUTION.md
```

`verify_raw.py` 不在计划原表里，实际是必需的：它既是 T2 的出闸门，也是 T3 的入口守卫，
沿用 `scale/pull_corpus.py` 与 `scale/verify_corpus.py` 的成对体例（pull 冻结、verify 守卫）。

---

## §15 命令速查（按执行顺序）

```bash
# ── 本地：T0 改造与验收 ─────────────────────────────────
.\.venv\Scripts\python.exe -m pytest ModelLakeFishing/stage2TrainGraphSAGE/tests -k "fanout or sparse_M or chunked"
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.export_ours `
  --graph .../hgraph_ml_v2_sub.pt --seed 0 --epochs 25 --tag R0_anchor_post
#   ⇒ 锚点判据见 G-C2

# ── watGPU：T1 探测 ──────────────────────────────────
sinfo -o "%P %N %G %m %c %l"; quota -s
srun --gres=gpu:1 --time=00:05:00 --pty bash -c 'curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models?limit=1'
git pull --ff-only
git rev-parse HEAD; git status --short

# T2–T8 的 python 命令写入对应的 sbatch wrapper，由 srun 在计算节点执行。
# 登录节点只负责同步、轻量检查、提交与监控。

# ── T2 v2（已完成，本地，2026-08-03）─────────────────
# HF_TOKEN 可选（匿名限流 500 req / 300 s，limit=1000 每页）
C=$MLF_DATA_DIR/data1m/candidates_v2
python -m scale1m.hf_crawl --plan full                              # 359,388
python -m scale1m.hf_crawl --plan deep --out $C                     # 492,391
python -m scale1m.hf_crawl --backfill-deficits $C/selected/DEFICITS.json --scale 0.3 --out $C   # 537,025
python -m scale1m.annotate_candidates --candidates $C
python -m scale1m.select_balanced_halo --annotated $C/annotated.parquet --target 69817 \
       --exclude-core stage1BuildTransferGraph/hgraph_ml_v2.pt      # D-35
python -m scale1m.select_balanced_halo --annotated $C/annotated.parquet --target 69817 \
       --exclude-core stage1BuildTransferGraph/hgraph_ml_v2.pt --out $C/selected_rerun --quiet   # G19
python -m scale1m.verify_raw --selected $C/selected/selected_halo.parquet \
       --annotated $C/annotated.parquet --rerun-check $C/selected_rerun/selected_halo.parquet    # G1-G20
python -m scale1m.report_distribution --v1 .../annotated_v1.parquet --pool $C/annotated.parquet \
       --selected $C/selected/selected_halo.parquet --out docs/1M/T2_runs --tag v2
python -m pytest scale1m/tests -q

# v1 快照守卫（raw/ 仍只读、仍可验）
python -m scale1m.verify_raw --core stage1BuildTransferGraph/hgraph_ml_v2.pt --sample 20 --forecast 100000

# ── T3 梯子（已完成，本地 CPU，2026-08-03）──────────
python -m scale1m.hf_canonicalize --candidates $C --core .../hgraph_ml_v2.pt   # 约 3 分钟
python -m scale1m.build_ladder --rung 100k --n 100000 --core .../hgraph_ml_v2.pt \
       --halo $C/selected/selected_halo.parquet --canon $C/canon/hf_canon.parquet \
       --out .../ladder                                                        # 约 20 秒

# ── T4 特征（已完成，watGPU）─────────────────────────
sbatch --export=ALL,RUN_ID=T4_100k_$(date -u +%Y%m%dT%H%M%SZ) scripts/watgpu/embed_lake.sbatch

# ── T5 建图与验图 ───────────────────────────────────
python -m scale1m.build_graph_rung --rung 100k --core ... --ladder ... --feats ... \
       --canon $C/canon/hf_canon.parquet --out .../hgraph_100k.pt
python -m scale1m.verify_rung_graph --rung 100k          # 全绿才往下

# ── T6 训练（三步）──────────────────────────────────
RUNG=12k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # 锚点
RUNG=30k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # D-9 解除
RUNG=100k SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh   # 冒烟
RUNG=100k SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # 全量

# ── T7 导出与索引 ───────────────────────────────────
python -m scale1m.export_rung --rung 100k --run $OUT --chunk 50000 --hnsw-threads 8

# ── T8 评测 ─────────────────────────────────────────
python -m scale1m.eval_rung --rung 100k --run $OUT --expect-n 100000 --axes a,b,c,d

# ── 监控与验收 ──────────────────────────────────────
squeue -u "$USER"
scontrol show job -dd <JOB_ID>
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode

# ── 本地回收（WSL/Linux/macOS）────────────────────────
rsync -avh --partial --info=progress2 \
  watgpu:"<REMOTE_OUTPUT_ROOT>/runs/<RUN_ID>/" \
  "<LOCAL_RESULTS_PATH>/<RUN_ID>/"
```

---

## §16 本档决策登记（续 `plan.md` D-17）

| ID | 决策 | 定案 | 理由 |
|---|---|---|---|
| D-18 | HALO 用哪个描述符函数 | CORE 的 `modellens_build_graph.model_descriptor(mid, family, size_b)`，HF 富字段不用 | 信息量需要对齐，否则两群平凡可分、`gold@10` 的不降是假的。报告须写明「主动放弃 HF 更丰富的元数据是受控实验的要求」 |
| D-19 | CORE 特征是否重算 | 不重算，原样复制 | MiniLM 跨设备的末位差异会让 CORE 不再是 P3/P5 的 CORE，五档梯子无法对比 |
| D-20 | 爬取是否一次到 1M | 否。按 downloads 降序、可续、只追加 | 产物天然是前缀嵌套，R3/R4 续爬即可。补充：游标是建在可变字段 `downloads` 上的 keyset cursor，跨周续爬只是近似嵌套，因此按 id 去重是必需的，每档 `snapshot_date_utc` 分别披露 |
| D-21 | `plan.md §4` 的第 5/9/10 项 | 100K 档不做，留给 R3/R4 | 数据集数不变、fp32 装得下、图约 250 MB，不提前优化 |
| D-22 | R2 的 seed 数 | 3 seeds（0,1,2） | 100K 单次约 30 分钟，没有省的理由；R3/R4 才考虑降到 1 |
| D-23 | watGPU 路径与存储 | 集中配置，首次登录实测后填写 | 集群存储、quota 与授权目录不能由本地路径或通用集群惯例推断 |
| D-24 | 计算执行位置 | 登录节点仅做轻量控制；冒烟用短时 allocation；正式运行用 `sbatch` | 资源密集阶段由 Slurm 调度并留下资源与作业记录 |
| D-25 | preemption/requeue 身份 | 沿用 logical `RUN_ID`，恢复完整训练状态；新实验分配新 `RUN_ID` | Slurm job ID 可能变化，实验身份与科学记录需要保持连续 |
| D-26 | HALO 的 `size_b` 是否用名字正则兜底 | 不用，只取 `safetensors.total` | 覆盖率差异本身是一条文本可分性通道。v2 实测 HALO 57.75% 对 CORE 47.91%（+9.83 pp），加正则会推到约 78.60%（偏离更远）。主动放弃 62,813 条可从名字恢复的参数量，报告须写明 |
| D-27 | `dropped` 层判据恒为假怎么办 | 保留判据不动，报告里写明它恒为 0 | HF 自动挂 `region:*`/`license:*` 标签，`tags` 覆盖率 100%，合取式永假。`layer` 只进审计不进训练（D-10），恒空无下游影响；改判据等于引入一条未验证的新过滤规则 |
| D-28 | 爬到的 1,199 个 CORE 模型的 `base_model` 用不用 | 不用，不给 CORE 子图加任何边 | 用了就是改动 CORE，R0/R1/R2 之间「N 是唯一变量」不再成立。这是明确的不作为，不是遗漏 |
| D-29 | T2 是否必须在 watGPU 上跑 | 否，本地跑并上传 | 产物 24.8 MB、46.8 秒，传输是秒级。D-12 的外网分叉保留为 R4 的优化项，不再阻塞 T1/T2 |
| D-30 | HALO 总体怎么定义 | 不是「downloads 榜前 N」。爬取与选样分离：多源发现 → 标注 → 家族/近重复归并 → 配额分配 → 确定性选样 → 缺口回补 → 冻结。全局 downloads 排名只作为候选发现的一路信号与分层内的流行度信号 | v1 实测：一个发布者 24.21%、量化搬运 61.98%、text-generation 50.77%、任务熵 0.601，是采样机制的产物而非样本量问题。v2 后同口径为 0.90% / 15.00% / 18.00% / 0.918 |
| D-31 | 百分比上限的分母 | 实际选出的总体，不是 requested target。上限按工作规模 N 参数化，二分搜索满足 `achieved(N) ≥ N` 的最大 N；出闸门再按真实计数独立复核 | 用 target 当分母会让「把 target 写大」买到更宽的上限，且跑不满时按实际产出算已越界。初版的向下自适应循环还会把可行问题误判为不可行（69,817 降到 60,202） |
| D-32 | 「英语侧 ≥75%」的分母 | 可归属语种的总体（除 `language-neutral` 外的全部）。`language-neutral` 另有自己的全体份额上限 | 把 ViT 计入英语份额不测量任何东西，且与任务均衡冲突。实测按全体作分母时卡在约 60,200，改分母后 69,817 一次探测即可行 |
| D-33 | 官方量化与第三方搬运 | 分成两类 source type（判据是量化仓库作者是否等于 base 作者）。15% 硬顶落在第三方类，两者合计另设 20% 硬顶 | `Qwen/Qwen3-8B-GGUF` 与 `somebody/Qwen3-8B-i1-GGUF` 不是同一类总体；合计上限防止用拆分抬高总量 |
| D-34 | 任务份额是目标还是硬顶 | 两者都要：15% 是配额分配目标，18% 是任何阶段的硬顶，G8 按实际总体检查 | 只有目标则放松阶段无界；只有硬顶则配额分配失去梯度 |
| D-35 | 选样前是否剔除 CORE | 必须剔除（`--exclude-core`） | 候选池按任务与语种发现会自然捞到 CORE 成员（池中 7,372 条）。不剔除时 69,817 条里有 1,892 条已在 CORE，`CORE ∪ HALO` 只有 98,108 个 distinct，而 R2 的定义是 100,000，且 G1–G20 都不会报警。剔除后实测交集 0、并集 100,000 |
| D-36 | HALO 的 `family` 怎么推 | 四级解析，目标是 CORE 的 `config.model_type` 命名空间；每行记 `family_source` | 规则函数 `_infer_one_family` 只得 0.64% 的 CORE-used 命中（返回 `LLaMA`，CORE 用的是 `llama`，vocab 里是两行），同架构占两行嵌入且不报错。四级解析升到 48.47%。附带纠正：该函数没有 tags 兜底 |
| D-37 | `lineage_base` 取哪个字段 | `baseModels` 优先，`cardData.base_model` 兜底，并保留 `relation` | 537K 候选上 `baseModels` 命中 234,445、`cardData.base_model` 只有 12,713，只用后者会丢 94.9%。`relation` 正是 `r_mm'` 离散权重需要的量 |
| D-38 | 梯子的 CORE 侧填不填四个量 | 留空，`layer="core"` | 约束 1 要求 CORE 的这些量从冻结图原样复制。梯子在 CORE 侧只钉 id 与行序，T4 直接读 `hgraph_ml_v2.pt` |
| D-39 | G-B4 没过门怎么处理 | 不回滚描述符。改为报三个数：绝对可分性 0.9729、流水线增量 0.118、只用名字的基线 0.8472 | 能调的只有流水线那 0.118，且已调到最接近 CORE 的一档（D-26）；剩下的 0.847 是两群构成本身的差异，改描述符改不动，改选样要推翻 T2 的 G1–G20。下游约束见 §7.3 |
| D-40 | family 子句覆盖率差是否列入归因 | 列入，且排在第一位 | CORE 只有 66.02% 的描述符带 family 子句，HALO 是 100%，差 34 pp，比 size 子句的 9.83 pp 大三倍多，机制相同 |
| D-42 | §9.1 列的 `--amp` 做不做 | R2 不做，推到 R3/R4 显存不够时再加，且加时单独跑一次锚点对拍 | 同一批改动里既换环境又换数值精度，锚点一偏就无法归因。实测 12K 峰值 0.583 GB、100K 峰值 1.737 GB，显存目前不是约束 |
| D-43 | `--seed` 的语义与锚点噪声带 | `--seed` 是 `split_seed`，`init_seed` 恒为 0（沿用 `export_ours.py`）。多 seed 给的是划分噪声区间，报数字时写明是哪一种 | 三个 100K seed 的 held-out 数据集数是 734 / 775 / 936，`tau_macro` 0.180 / 0.077 / 0.217，测试集本就不同。G-C2 引用的「0.4159 ± 3-seed 噪声」中的噪声带从未被测过，`P3_EXECUTION.md` 的 0.4159 是单次结果 |
| D-44 | 100K 图的血缘 `relation_id` 全是 `unknown` 怎么办 | 重建：补传 `hf_canon.parquet`，重建 100K 图，只重跑 100K 档的四个 run。旧图与三份报告归档为 `hgraph_100k_norel_7fbc3c47.pt` 与 `norel_*.json`，8-11 那套证据保持可复验；12k/30k 不动 | `hf_canon.parquet` 从未传到远端，D-37 拿回的 94.9% 血缘声明没有进图，`r_mm'` 的离散有序权重无输入。逐张量比对证明重建干净：五种边的 `edge_index`/`edge_attr` 与 `model.x`/`size_bucket_id`/`family_id`/`dataset.x` 全部相同，唯一不同的是 `relation_id` |
| D-45 | `--time` 申请多大 | 按实测定，不留大额占位值：`train_rung.sbatch` 默认从 `04:00:00` 改为 `00:30:00`（约为实测最慢 2m44s 的 7 倍）；R3/R4 在命令行上抬高 | Slurm 按申请的墙钟排队。4 小时的占位值让六个两分钟的作业被排成每 4 小时一个，最后一个排到次日 13:17；改为 00:20:00 后压缩进同一个晚上，节省约 14 小时 |

待裁定的一条：G-C2 的判据是否从「复现 0.4159 这个点值」改写为「12K 的 3-seed 区间覆盖历史值」。
按后一口径，12K `approx` 区间 [0.349, 0.387] 仍不覆盖 0.4159，而 `exact` 区间 [0.340, 0.468] 覆盖它。
T7.5 建议改写并另立 R2 基线，但这需要单独定案。

---

## §17 与 `plan.md` 的对应关系

| `plan.md` 阶段 | 本 runbook | 100K 档的裁剪 |
|---|---|---|
| S0 环境 | T1 | access、存储、同步、环境、smoke、tiny `sbatch` 全部过门；`--mem 96G`，`--time` 由实测修正为 00:30:00（D-45）。不再是 T2 的前置（D-29） |
| S1 数据 | T2–T5 | 图不拆件（D-21）。T2 完成 v2（532 个查询 → 537,025 候选 → 配额选出 69,817，18/20 硬门过，与 CORE 零交集，v1 的 15 万条降级为审计件）。T3 完成（梯子 100,000 行，五门全过）。T4 完成（四门过，G-B4 按 D-39 改判）。T5 完成并按 D-44 重建 |
| S2 代码 | T0 | 10 项做 7 项（D-21）；出闸门 G-A1 未通过、G-A4 由 T7.5 补做 |
| S3 作业化 | T1 + T6 | 训练与建索引不拆作业（100K 的索引 CPU 开销还小）；batch metadata、checkpoint/resume、监控与回收形成闭环 |
| S4 执行 | T6 | R0 → R1 → R2 三档全部跑完 |
| S5 评测 | T8 | A/B/C/D 四轴全做；曲线拟合只有 3 点，不下结论，等 R3/R4 |

---

## §18 变更记录

只记录会改变读者理解的口径变化，细节见各执行记录。

| 日期 | 变更 |
|---|---|
| 2026-08-02 | T2 在本地跑通，产物 24.8 MB / 46.8 秒。外网访问从 T2 的前置条件降级为 R4 的优化项（D-29），阶段依赖图相应调整 |
| 2026-08-03 | HALO 的定义从「downloads 榜前 69,817」改为配额均衡选样的总体（D-30）。所有按 v1 前缀算出的预测（layer 分布、size 覆盖、血缘边数、frozen 占比）作废，由 T3 用 v2 总体重算 |
| 2026-08-03 | 血缘边的预测从 26,297 改为 16,385（`core_halo` 606、`halo_halo` 15,779），原因是近重复上限削掉了血缘最密的一群。T5 实测与该预测一致 |
| 2026-08-03 | `frozen` 层的占比从 43.6% 改为 53.48%（预测），T8 实测 50.01%，C 轴的重心从次要问题变成过半总体的问题 |
| 2026-08-10 | T4 实测 G-B4 未过原门（AUC 0.9729），改判为双口径（D-39），并把 family 子句覆盖率差补进归因清单第一位（D-40） |
| 2026-08-13 | 100K 图重建，补上血缘 `relation_id`（D-44）；`--time` 从 04:00:00 改为 00:30:00（D-45） |
| 2026-08-14 | T7 首次执行锚点门并未通过（G-C2）；补齐 `git_head`、`family_vocab` 绑定与 as-run 代码对账 |
| 2026-08-14 | T7.5 用三档 × 三臂 × 三 seed 补做 D-14 要求的对拍（G-A4），排除对比损失采样与扇出上限作为 G-C2 的原因，并给出「A 轴数字必须带 seed 区间」的纪律 |
