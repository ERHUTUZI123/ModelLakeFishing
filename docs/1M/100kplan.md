# R2 = 100K 档完整执行流程（Runbook）

**上级规划：** [`plan.md`](plan.md)（五档梯子 R0–R4、双层湖设计、S0–S5 阶段划分）
**本档定位：** 梯子的**第一个真实放大档**。它同时是后面 R3/R4 的**全部基础设施的首次端到端验证** —— 100K 跑通，500K/1M 就只是换个 `--limit` 和 `--mem`。
**目标 N：** 100,000 模型（CORE 30,183 + HALO 69,817）
**算力：** watGPU A100/H100 单卡
**预计墙钟：** 代码改造 3–5 天（本地，可并行）+ 数据 1 天 + 训练评测 1 天

---

## §0 这一档必须交付什么

| # | 交付物 | 验收 |
|---|---|---|
| D1 | `hgraph_100k.pt`（100,000 模型 / 9,603 数据集节点 / CORE 监督冻结） | §12 G-B 全过 |
| D2 | R0(12K) 与 R1(30K) 的**锚点复现**数字 | `gold@10(R0) = 0.4159 ± 3-seed 噪声` |
| D3 | R2 三轴数字（A 精度 / B iso-recall 延迟 / C 冷启动分层） | §12 G-D 全过 |
| D4 | `z_m [100000,128]` + `hnsw_100k.bin` + `MANIFEST.json` | 行序断言通过 |
| D5 | `docs/1M/S4/R2_EXECUTION.md`（含**失败过程留痕**，沿用 P0–P5 体例） | — |
| D6 | 可直接复用于 R3/R4 的爬虫/建图/训练/评测代码 | R3 只需改 `--limit` |

**这一档不交付的**（明确排除，防止范围蔓延）：ModelLens 对照实测（D-16 定：外推）、HALO 标签进监督（D-10 定：不进）、500K/1M 的任何数字。

---

## §1 三条铁律（违反任何一条，这一档的数字全部作废）

### 🔴 铁律 1：CORE 逐字节冻结

CORE 的 30,183 行特征、size/family id、`trained_on` 边、`unique_model_id` 表，**从 `hgraph_ml_v2.pt` 原样复制**，**绝不重新计算、绝不重新嵌入**。

理由：MiniLM 在不同设备/不同 batch 组成下末位有浮点差异；重嵌一次，CORE 就不再是 P3/P5 那个 CORE，五档梯子和全部历史战报当场断链。

```python
# 建图末尾必须有的断言
core = torch.load("hgraph_ml_v2.pt", weights_only=False)
assert torch.equal(data["model"].x[:30183], core["data"]["model"].x)
assert torch.equal(data["model"].size_bucket_id[:30183], core["data"]["model"].size_bucket_id)
assert torch.equal(data["model"].family_id[:30183],     core["data"]["model"].family_id)
assert umi.iloc[:30183]["model"].tolist() == core["unique_model_id"].sort_values("mappedID")["model"].tolist()
assert torch.equal(data["dataset"].x, core["data"]["dataset"].x)          # 数据集侧完全不动
assert torch.equal(data[TRAINED_ON].edge_index, core["data"][TRAINED_ON].edge_index)
```

### 🔴 铁律 2：HALO 必须用**与 CORE 同一个**描述符函数

**这是最容易犯、且犯了看不出来的错。** 仓库里有两个同名不同义的函数：

| 函数 | 位置 | 产出文本 |
|---|---|---|
| ✅ **CORE 用的** | [`scale/modellens_build_graph.py:67`](../../scale/modellens_build_graph.py#L67) `model_descriptor(mid, family, size_b)` | `"bert base uncased family bert 0.11B params"` |
| ❌ **不许给 HALO 用** | [`stage1BuildTransferGraph/d0_build_graph.py:102`](../../stage1BuildTransferGraph/d0_build_graph.py#L102) `model_descriptor(d, mid)` | `"bert base uncased fill-mask transformers pytorch en datasets: bookcorpus wikipedia"` |

**如果 HALO 用了 HF 富字段版**：两群模型的 `e_desc` 文本分布系统性不同（长度、词表、句法全不一样）⇒ MiniLM 空间里 CORE 与 HALO 天然可分 ⇒ **HALO 永远挤不掉 CORE 的 gold** ⇒ `gold@10` 几乎不降 ⇒ 得到一个**看起来很棒但完全是假的**「扛住了 83× 干扰」的结论。

**做法**：从 HF 元数据里只抽 `family` 和 `size_b` 两个量，喂进 **CORE 那个** `model_descriptor`。HF 的 tags / pipeline_tag / library_name **一律丢弃**（除了用来推 family）。

> **D-18（本档新增决策）**：HALO 描述符 = CORE 描述符函数 + 从 HF 推出的 (family, size_b)。**信息量对齐优先于信息量最大化。**
> 副作用要诚实承认：我们**主动放弃**了 HF 上更丰富的模型元数据。这不是能力上限，是受控实验的要求。**在报告里明写这一句。**

> 🔴 **T2 实测补充（F-T2-3）：同一个函数 ≠ 同样的信息量。**
> `model_descriptor` 在 `size_b=NaN` 时**整段省略** `"<x>B params"` 子句 ⇒ 「多大比例带 size 子句」本身就是一条
> 系统性文本差异，**铁律 2 的字面表述管不到它**。实测 CORE 47.91% 带 size 子句，HALO 只用 `safetensors` 是 38.17%，
> 加名字正则会变成 78.60%。⇒ **D-26 定案：只用 `safetensors.total`，不加正则兜底**（差 −9.7 pp，三个选项里最接近 CORE）。
> **铁律 2 应读作：同一个描述符函数 + 尽量对齐的字段覆盖率。G-B4 的 AUC 是这一层唯一的守门员。**

### 🔴 铁律 3：评测时候选池必须真的是 100,000

最可能的静默 bug：评测代码只对 CORE 的 30,183 个模型打分，`gold@10` 于是纹丝不动，而你以为「扛住了」。

```python
# five_metric_eval / global_metrics 里必须有
assert z_m.shape[0] == args.expect_n, f"candidate pool is {z_m.shape[0]}, expected {args.expect_n}"
```
并在 `MANIFEST.json` 里回写实际 `N_candidates`。**§12 的 G-D1 就是查这个。**

---

## §2 阶段依赖图

```
        ┌─ T0 代码改造（本地，12K 上验收）──────────┐
        │   不依赖任何外部条件，今天就能开工          │
        │                                            ▼
        ├─ T2 HF 爬取 ──> T3 梯子 ──> T4 特征 ──> T5 建图 ──┐
        │  （本地即可，产物 25 MB 可移植）                   ▼
T1 watGPU 环境 ───────────────────────────────> T6 训练 ──> T7 导出+索引 ──> T8 评测
                                                     ▲
                                            T6 前必须 T0 全绿
```

**关键路径 = T1 ∥ (T2 → T3 → T4 → T5) → T6 → T7 → T8。T0 与 T2 都不依赖 T1，先开这两个。**

> **T2 实测后修正（`T2.md` F-T2-12）**：原图把 T2 挂在 T1 之后，理由是「计算节点有无外网决定 T2 在哪跑」。
> 实测 T2 产物只有 **24.8 MB**、墙钟 **46.8 秒**，本地跑完 `scp` 上传是秒级。
> **外网分叉（D-12）保留为 R4 的优化项，但不再是 T2 的前置条件。** T2 已于 2026-08-02 在本地跑完。

---

## §3 T0：代码改造（本地，在 12K 图上验收）

> 100K 档需要 `plan.md §4` 十项里的 **7 项**。第 5、9、10 项 100K 用不上（数据集数不变、fp32 装得下、图文件 ~250MB），**留到 R3/R4 再做**。

### T0.1 建基线（改造前先钉住对照组）

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only v2      # 必须 22/22 全绿
# 记录当前 12K 基线（若 P3 产物还在可直接引用，否则重跑一次）
Set-Location D:\research\model_lake\codes
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.export_ours `
  --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt `
  --seed 0 --epochs 25 --tag R0_baseline_pre
```
**记下 `gold@10`（应为 0.4159）。这是后面每一项改造的锚点。**

### T0.2 改造项（按此顺序，每项改完立刻跑锚点）

#### ① 🔴 扇出上限 loader — `stage2TrainGraphSAGE/sampling.py`

- **改哪**：`LightLinkLoader._khop_closure`（[sampling.py:88](../../stage2TrainGraphSAGE/sampling.py#L88)）
- **现状**：全邻域闭包，且用 `torch.isin(src, tensor(sorted(cur)))` 全表扫描
- **改法**：
  ```python
  def __init__(self, ..., num_neighbors=(10, 10), gen=None):
      self.fanout = list(num_neighbors)
      # 预建 CSR：按 src 排序 + ptr，邻居查找从 O(E) 降到 O(log N + deg)
      self._csr = {et: _build_csr(ei, n_src) for et, ei in self._edges.items()}

  def _khop_closure(self, seed_m, seed_d):
      # 每跳：对每个 frontier 节点，从其邻居里随机抽 min(deg, fanout[hop]) 个
  ```
- **参数**：`num_neighbors=(10,10)`；血缘关系用**更大的扇出**（`(20,20)`），因为它是冷启动模型唯一的边（CLAUDE.md Step 4 明写"lineage 自己的、更低的丢弃率"，同理这里给更高的保留率）
- **验收**：
  - 单测 `test_fanout_bounded`：fanout=(2,2)、batch=128 ⇒ 子图模型节点数 ≤ 128×(1+2+4) 的可证上界
  - 锚点：12K 上 `gold@10` 与 T0.1 在 3 seeds 内统计不可区分

#### ② 🔴 稀疏 `M` — `stage2TrainGraphSAGE/losses.py`

- **改哪**：[`topk_membership`:415](../../stage2TrainGraphSAGE/losses.py#L415)、[`pool_membership_by_root`:426](../../stage2TrainGraphSAGE/losses.py#L426)、[`global_positive_density`:467](../../stage2TrainGraphSAGE/losses.py#L467)、`dataset_to_model_contrastive` 里的 `M[cand, d]`
- **数量级**：100K × 9,603 稠密 = **3.84 GB**（1M 时 38.4 GB）；非零元只有 ~31K，**稠密度 3×10⁻⁵**
- **改法**：`M` 改 `torch.sparse_csc_tensor`；`pool_membership_by_root` 的 python 循环换 `scatter_max` 按 root 聚合
- **验收**：`test_sparse_M_equivalence` —— 12K 上稀疏版 `.to_dense()` 与原稠密版**逐元素相等**；`pool_membership_by_root` 输出相同

#### ③ 🔴 对比损失采样负例 — `losses.py:498` `contrastive_loss`

- **数量级**：即使扇出上限把子图压到 20K 节点，`[20000,20000] × 4 张 × 4B = 6.4 GB`
- **改法（A 方案）**：每锚点抽 `n_neg` 个负例（默认 256，与 L1L3b 的 `global_n_neg` 同源），复杂度 O(B×n_neg)
- **🔴 D-14 强制要求**：这是**改动冠军配置**。必须做 A/B 对拍并写进报告：

  | 配置 | seeds | gold@10 mean ± bootstrap CI |
  |---|---|---|
  | 原版全 N² 对比损失 | 0,1,2 | （基线） |
  | 采样负例 n_neg=256 | 0,1,2 | ？ |
  | 采样负例 n_neg=1024 | 0,1,2 | ？（若 256 显著更差则试） |

  **若采样版显著更差**：改用 B 方案（分块 logsumexp，数学等价、只省显存、时间仍 O(N²)），或把 100K 的子图上限压得更小。**不许默默接受退化。**

#### ④ 🔴 分块推理 — `scale/export_ours.py:122,135`

- **现状**：`model(data.clone().to(device))` —— 整图搬 GPU
- **100K 上**：`x[100K,448]` = 179 MB，2 层 hidden × 5 种关系的中间量 ⇒ 峰值预估 **4–8 GB**。A100 装得下，**但 R3/R4 装不下，且这段代码不改 R4 必死**。100K 就是它的首次实战验证。
- **改法**：预分配 `z_m = torch.empty(N,128)`，用扇出 loader 按 `batch_size=50_000` 遍历模型节点，`torch.no_grad()` 逐块写入
- **🔴 行序**：写入必须 `z_m[batch['model'].n_id] = out`，**绝不能 `z_m[i*B:(i+1)*B] = out`**。CLAUDE.md 把行序错位列为三大风险之首，这里是它最容易发生的一行。
- **验收**：12K 上分块推理结果 vs 全图前向 **逐元素 `max|Δ| < 1e-5`**

#### ⑤ 🔴 流式打分 — `top1_eval.py:68` / `scale/global_metrics.py`

- **现状**：[export_ours.py:128](../../scale/export_ours.py#L128) 把 517 个查询 × N 的完整分数向量物化进一个 dict
- **100K 上**：517 × 100K × 4B = **207 MB**（能扛，但 1M 时 2 GB 且 CPU matmul 极慢）
- **改法**：打分搬 GPU，`z_m` 常驻显存，逐查询/分块算 `z_d_q @ z_m.T`，**只保留 gold rank、gap rank、top-K id**，不保留完整向量
- **🔴 验收**：P3 验证过的「`five_metric_eval.gold@10 == global_metrics.gold@10` 逐查询对账」**改完必须重跑并仍然相等**

#### ⑥ 🟠 诊断可关 — `losses.py:467`

- 加 `--skip-diagnostics`；100K 档默认**开着**（稀疏化后不再是瓶颈），R3/R4 默认关

#### ⑦ 🟠 HNSW 多线程 + 落盘 + iso-recall — `export_ours.py:58`

- `init_index(max_elements=N)` 已参数化；新增 `--hnsw-threads`、`--ef-construction`、`--M`
- `idx.save_index()` 落盘，避免评测时重建
- **新增 `tune_ef_for_recall()`**：二分 `ef_search` 使 `recall@50 ≥ 0.99`，返回最小可行 ef。**延迟必须在这个 ef 下测**（§11.2）

### T0.3 T0 出闸门

- [ ] 7 项全部改完，各自单测通过
- [ ] 🔴 **锚点门**：12K 上 3 seeds，`gold@10` 与 T0.1 基线 **bootstrap CI 重叠**
- [ ] 分块推理 vs 全图前向 `max|Δ| < 1e-5`
- [ ] `five_metric_eval == global_metrics` 逐查询对账仍成立
- [ ] D-14 的 A/B 对拍表已产出（无论结论好坏）

> **T0 不过，不许碰 watGPU。** 在错的代码上烧 A100 小时是纯浪费。

---

## §4 T1：watGPU 环境（详见 `plan.md §2`，这里只列 100K 档的增量）

执行 `plan.md §S0.1` 的五组探测命令 → `docs/1M/S0/WATGPU_ENV.md`。

### 4.1 执行边界与集中配置

watGPU 登录节点仅承担 Git 操作、环境管理、轻量文件检查、Slurm 提交与监控。预处理、批量下载、模型加载、训练、评测及 GPU 检查均在 Slurm 分配的计算节点执行。交互式 allocation 仅用于短时调试和验证；正式运行统一使用 `sbatch`。本地 SSH 断开不影响已提交的 batch job。

路径和资源参数集中写入未含密钥的配置文件或作业环境。仓库仅提交占位模板 `configs/watgpu.example.env`；含账号、token 或实际路径的文件不得进入 Git。

| 变量 | 含义 | T1 要求 |
|---|---|---|
| `PROJECT_ROOT` | 远端代码目录 | 登录后确认，不写入本地 Windows 绝对路径 |
| `DATA_ROOT` | 数据、图与只读快照目录 | 先确认容量、quota 与清理策略 |
| `OUTPUT_ROOT` | `runs/`、checkpoint、metrics 与日志目录 | 与代码目录解耦，保证可续跑 |
| `CACHE_ROOT` | Hugging Face、Torch 与 pip cache | 显式设置 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR` |
| `ENV_ACTIVATE` | venv/conda 激活命令 | 以 `plan.md §S0.2` 的实测结果为准 |
| `PARTITION` / `SLURM_ACCOUNT` | 分区与 account | 由实时权限探测填写 |
| `GPU_CONSTRAINT` | 可选 GPU constraint | 默认留空；仅在实测表明确有必要且集群支持时填写 |

`$WORK` 仅在实时探测确认后作为上述根目录的来源。**不得预设 `/scratch`、`/project` 或其他大容量文件系统存在。**

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

`WATGPU_ENV.md` 必须记录 `HOME_DIRECTORY`、`PROJECT_DIRECTORY`、`DATA_DIRECTORY`、`OUTPUT_DIRECTORY`、home quota、大文件目录、授权 partition/account、可用 GPU constraint、Python 与环境管理器版本。存储位置或配额未确认时，暂停大数据上传。GPU 型号、driver、CUDA runtime 与显存只在计算节点 allocation 内记录。

### 4.3 代码、数据与环境同步

- 代码默认通过 private Git repository 同步。每次正式运行前执行 `git pull --ff-only`，记录 `git rev-parse HEAD` 与 `git status --short`。生产作业通常使用 clean commit；有意保留的未提交变更写入运行目录的 `metadata/uncommitted.patch`。
- 大文件重复传输优先使用 `rsync --partial`；只追加数据可使用 `--append-verify`。`--delete` 不进入默认命令。CORE 图、corpus 与关键分片在源端和远端校验 SHA-256。
- 远端环境由锁定的 requirement/environment specification 重建。PyTorch/CUDA 组合依据计算节点 driver 与官方兼容性确定，不直接复制本地 Windows wheel 选择。安装后保存 `pip freeze` 或等价环境记录，并执行 `python -m pip check`。
- dataset、checkpoint、cache、log、token、SSH key 与私有 `.env` 不进入 Git。HF token 通过作业环境注入，不写入脚本和 Slurm 日志。
- 数据上传前确认 privacy、license、ethics 与 residency 限制。受限数据放入非共享目录，并使用 `umask 077` 或等价权限控制。

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

项目冒烟必须完成 import、一个 tiny batch 的 forward/backward、checkpoint save/reload/resume、一个既有评测指标及输出路径检查。仓库和 resolved config 需确认不存在 `C:\\` 或 `D:\\` 路径。验证结束后退出计算节点并释放 allocation。

### 4.5 100K 资源与外网分叉

**100K 档特有的资源需求：**

| 项 | 需求 | 备注 |
|---|---|---|
| GPU | **1× GPU，显存 ≥ 24 GB** | A100/H100 均可；提交时默认不限定型号 |
| `--mem` | **96G** | 峰值在建 HNSW 与稀疏 M 构造 |
| `--cpus-per-task` | 8 | HNSW 建索引多线程 |
| `--time` | **04:00:00** | 先用 `--epochs 1` 实测单 epoch，再按实测修正 |
| 磁盘 | ~1 GB | ~~爬取原始 3 GB~~ **爬取 0.025 GB（实测）** + 特征 0.2 GB + 图 0.25 GB + 索引 0.1 GB + 运行目录 |

**🟠 一个曾被列为"决定性"的分叉**（`plan.md` D-12），T2 实测后**降级**：

```bash
srun --partition=<P> --gres=gpu:1 --time=00:05:00 --pty bash -c \
  'curl -sS -m 10 -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models?limit=1'
```

- 返回 `200` ⇒ T2 **可以**在 watGPU 上跑（R4 的 ~200 MB 值得就近爬）
- 超时/拒绝 ⇒ T2 在本地跑，产物 `scp`/`rsync` 上传

> **T2 实测（`T2.md` §4 / F-T2-12）**：15 万条裁剪后 **24.8 MB**、150 个 HTTP 请求、**46.8 秒**、0 重试、**匿名无 token**。
> 计划原写「本地跑增加约 1 小时传输」—— 实际是秒级。**这个分叉不再阻塞任何东西**，探测照做（记进 `WATGPU_ENV.md`），但结论两条路都能走。
> R2 的 T2 已在本地完成，无需在 watGPU 重跑。

### 4.6 Batch 提交、监控与结果回收

提交 wrapper 在调用 `sbatch` 前创建 `logs/`；`scripts/watgpu/train_rung.sh` 启动后创建运行目录，启用 `set -euo pipefail`，并通过 `srun` 调用 T6 的原始训练命令。每次实验分配唯一的 logical `RUN_ID`；Slurm job ID 单独记录，requeue/续跑沿用原 `RUN_ID`。

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

监控采用人工检查或低频、有界轮询。`COMPLETED` 仅表示 Slurm 进程结束；科学验收还需确认最终 checkpoint、可解析 metrics、无 NaN/Inf、T8 完成、commit/config 对应正确及所有分片齐全。

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

## §5 T2：HF 候选发现与均衡 HALO 构建 ✅ **v2 已完成（2026-08-03，本地）**

> **执行记录：[`T2.md`](T2.md)。** 本节已按实测回填 —— 下面写的都是真跑出来的，不是估的。

### 🔴 5.0 v1 已作废为「总体定义」（D-30）

原设计（v1）= **按 `downloads` 降序爬前 15 万条，这 15 万就是 HALO**。
这一条技术上跑通了，**科学上不可用**：

| 观测量 | v1 前 15 万 |
|---|---|
| 最大发布者占比 | **24.21%**（`mradermacher`；前二名合计 32.18%） |
| 有效发布者数（1/HHI） | **15.1**（名义 25,500 个） |
| `quantized-conversion` 占比 | **61.98%** |
| `text-generation-chat` 占比 | **50.77%**（translation 0.61%，summarization 0.13%） |
| 任务分布归一化熵 | **0.601** |
| 元数据质量通过率 | 70.45% |

**这是采样机制的产物，不是样本量问题** —— 下载计数器奖励的是被自动化流水线反复拉取的次数，
再爬 100 万条只会让搬运仓库的绝对数变大。对下游的具体伤害（hub 过平滑、对比损失负样本是"同一个模型"、
跨任务检索无法测量、C 轴归因与 `source_type` 共线）见 `T2.md §1`。

**v2 的结构：爬取与选样分成两级。**

```
多源候选发现  →  标注  →  家族/近重复归并  →  配额分配  →  确定性选样  →  缺口回补  →  冻结
   （允许有偏）                                        （不允许有偏）
```

`raw/` 的 15 万条**原样保留、未被覆盖**，降级为审计件与候选源之一。

### 5.1 设计：发现阶段允许有偏，选样阶段不允许

发现阶段跑 **532 个查询**（全局 5 序 + 56 任务 × 2–3 序 + 34 语种 × 1–2 序 + 30 生态 + 10 结构 + 110 缺口回补），
共 1,799,891 次原始命中 → **537,025 条唯一候选**（重叠 3.35×，64.7 MB，~28 min）。
产物仍是可续、只追加的；R3/R4 从 `PLAN_STATE.json` 继续。

> 🔴 **F-T2-9：前缀嵌套只是"近似"，因为游标建在可变字段上。**
> `Link: rel="next"` 里的游标 base64 解出来是 `{"$or":[{"downloads":X,"_id":{"$gt":...}},{"downloads":{"$lt":X}}]}`
> —— keyset 建在 `downloads` 上，而 `downloads` 天天在变。R3/R4 隔周续爬时，期间涨过游标位的模型会被跳过。
> 两个强制后果：
> 1. **按规范化 id 去重是强制项**（已实现；R2 本次 0 重复，但那是 46.8 秒窗口的运气，不是保证）
> 2. **每一档的 `snapshot_date_utc` 分别记录、分别披露**，报告里不许写成"同一份 1M 快照的前缀"

> **F-T2-14 / F-T2-15（发现阶段的两条实测）**：
> ① 全局排序流之间边际收益衰减极快（`global:downloads` 新增 100%，`global:likes` 51.8%，`lastModified` 21.9%）
> ⇒ **覆盖来自按任务/语种切分，不是再加几种全局排序**。
> ② **浅层语种查询几乎无用** —— 各语种按下载量的前 240 条全是同一批多语大模型，新增为 0；
> 必须深挖（cap 6,000 + 按 `createdAt` 排序）才有增量。

### 5.2 `scale1m/hf_crawl.py`（已实现）

**🔴 F-T2-1（最重要的一条）：不能用 `full=True&cardData=True`，必须用 `expand[]`。**

| 路径 | 返回 `safetensors` | 返回 `siblings`（膨胀） | 后果 |
|---|---|---|---|
| ❌ `full=true&cardData=true` | **否** | 是（整个 repo 文件列表） | `size_b` **100% 缺失**，全 HALO 落 bucket 0，**不报错** |
| ✅ `expand[]=...` | **是** | 否 | 正确 |

`safetensors.total` 是 §6.1 里 `size_b` 的第一顺位来源，也是唯一可靠的来源。走错这条路，
T3–T8 全程照跑不误，只是 `e_size` 整群塌成一行，最后在 C 轴看到一个说不清的结果。

```
GET https://huggingface.co/api/models?limit=1000&sort=downloads&direction=-1
    &expand[]=downloads&expand[]=likes&expand[]=pipeline_tag&expand[]=library_name
    &expand[]=tags&expand[]=createdAt&expand[]=safetensors&expand[]=cardData
```

**要点（实现即如此）：**
- **token 可选，不是必须**（F-T2-2）：限流 `RateLimit-Policy: q=500;w=300`，即 **500 请求 / 300 秒**；
  `limit=1000` 是每页上限 ⇒ 15 万条 = **150 个请求**，一个窗口都用不满。仍保留 `--token`/`$HF_TOKEN` 与主动退避。
- **断点续爬**：`CURSOR.json`（`next_url` + 已写条数 + 分片内偏移 + stats）+ `SHARDS.json`（增量分片清单）。
  进程被杀在半行 ⇒ 续爬按 `CURSOR` 提交的条数截断（有单测）。
- **429 / 5xx 退避**：指数退避 + `Retry-After` + 限流余量 < 25 时主动睡到窗口重置；`retry_reasons` 全部记账。
- **字段裁剪**：`KEEP` 七个顶层 + `safetensors.total` + `cardData` 的 `base_model`/`datasets`/`model-index`；
  `model-index` 再压成 `(task, dataset, dataset_name, config, split, metrics[type,value])`，
  丢掉每条 metric 上数 KB 的 `verifyToken`。**~34 KB/条 → gz 后 165 B/条。**
- **不在爬虫里执行铁律 2**：`tags`/`pipeline_tag`/`library_name` 全量落盘。铁律 2 是 T3/T4 的纪律。
  理由：**快照不可重建** —— 今天为纪律少存的字段，明天要用只能重爬一份不同日期的快照，梯子当场断链。
- **产物**：`hf_models_*.jsonl.gz` + `SHARDS.json` + `CURSOR.json`/`PLAN_STATE.json` + `PROVENANCE.json`。
  分片与 `PROVENANCE.json` 均 `chmod 444`。

**🔴 v2 相对 v1 新增的六个字段（缺一不可，见 `T2.md §3.2`）**：
`author`（发布者集中度）、`baseModels`（**权威** parent + `relation`，`cardData.base_model` 只是手打字符串没有关系类型）、
`config`（`architectures`/`model_type`）、`lastModified`、`gguf`（权威量化标志）、`gated/disabled/private`。

> **F-T2-13：v1 的 `KEEP_CARD` 把 `cardData.language` 丢了** —— v1 的 15 万条里该字段数量为 **0**，
> 语种只能从 tags 的裸语种码猜，而裸 `^[a-z]{2,3}$` 会误收 `trl`/`sft`/`tf`/`jax`/`mms`。
> v2 补上该字段并改用 ISO-639 白名单。**因此 v1 的 40.88% "语种未知" 有一部分是爬取缺陷、不全是总体性质，
> 报告里必须一起说。**

> **F-T2-22：`library=` 参数被 HF 静默忽略**（`library=timm` 返回全局下载榜首，不是 timm 模型）。
> 必须用 `filter=timm`。不报错、不为空，只是悄悄不过滤。

### 5.3 标注与选样（v2 新增的两级）

```powershell
.\.venv\Scripts\python.exe -m scale1m.annotate_candidates  --candidates $C
.\.venv\Scripts\python.exe -m scale1m.select_balanced_halo --annotated $C\annotated.parquet --target 69817
```

**标注**（`taxonomy.py` + `annotate_candidates.py`）：18 类 supertask、5 类语种桶、7 类 source type、
量化方法/位宽、家族（血缘闭包，六级优先序）、近重复键、元数据质量分。
**每个标签都带 `*_source` 与置信度**，仓库名启发式一律自报家门、从不覆盖权威元数据。

**选样**（`select_balanced_halo.py`）：配额感知加权轮转，五轴同时跟踪，
再叠加发布者/搬运方/家族/近重复/top-10 五类硬上限。

> 🔴 **5.3.1 分母纪律（D-31，本档最重要的一次返工）**
> **所有百分比上限都是「最终选出总体」的份额，不是 `--target` 的份额。**
> 初版把 `text_gen ≤ 0.15 × requested_target` 写进选样过程，两个后果：
> ① **把 target 写大就能买到更宽的绝对上限**；② 跑不满 target 时，按实际产出算的份额已经越界。
> 修正做法：把上限按工作规模 `N` 参数化，**二分搜索满足 `achieved(N) ≥ N` 的最大 `N`**
>（`achieved` 随 N 单调不减、`achieved−N` 递减 ⇒ 判据单调 ⇒ 二分给出**精确**可行最大值），
> 在该不动点上每个上限按构造就是实际产出的正确份额；**出闸门再用真实计数独立复核一遍。**
> 单测 `test_caps_hold_as_shares_of_the_actual_population_not_the_request` 用同一池分别请求 400 / 4000，
> 断言两者都满足各自 1.5% 的份额。

> 🔴 **5.3.2 语种规则的分母（D-32）**
> 「英语侧 ≥ 75%」是关于**语种构成**的断言，分母是**可归属语种的总体**（除 `language-neutral` 外的全部）。
> 把 ViT 计入英语份额既不测量任何东西，又与任务均衡直接冲突：语种无关的 5 个 supertask
> 在任务配额下合计就要 >25%，与"非英语侧 ≤25%"互斥。
> **实测：按全体作分母时均衡 HALO 卡在 ~60,200 上不去；改用可归属分母后 69,817 一次探测即可行。**

> 🔴 **5.3.3 官方量化 vs 第三方搬运（D-33）**
> `Qwen/Qwen3-8B-GGUF`（第一方发布）与 `somebody/Qwen3-8B-i1-GGUF`（第三方重打包）不是同一类总体，
> 判据结构性可得：**量化仓库作者是否 == base model 作者**。15% 硬顶落在第三方类上，
> 两者合计另设 20% 硬顶，防止"拆分"被用来偷偷抬高总量。

> 🟠 **5.3.4 text-generation：15% 目标 / 18% 硬顶（D-34）**
> `max_share` 是配额分配用的目标，`hard_ceiling` 是任何阶段不可越过的硬顶，G8 按实际总体检查。

### 5.4 T2 出闸门 —— ✅ v1 快照门 + v2 总体门

**v1 快照门（`raw/` 仍然只读、仍然可验，作为审计件）**：

- [x] `PROVENANCE.json` 完整，`total_records = 150,000`；每片 sha256 复算一致（3/3）；分片只读（3/3）
- [x] 🔴 **快照日期已记录 ⇒ `2026-08-03 (UTC)`**（v1/v2 同日）
- [x] 规范化 id 去重后仍是 150,000（0 重复）；`downloads` 降序违例 = 0

**v2 总体门 G1–G20（`verify_raw --selected`，全部按实际选出的 69,817 计算）**：

**18/20 硬门通过；G14/G15 为联合约束所限并已定量披露。** 逐条见 `T2.md §6.5`，关键几条：

| ID | 判据 | 实测 |
|---|---|---|
| G3 | 普通最大发布者 ≤ 1.5% | **0.902%**（v1 是 24.21%） |
| G4 | top-10 发布者 ≤ 10% | **4.045%**（v1 是 40.42%） |
| G5 | 单个搬运发布者 ≤ 0.5% | **0.500%** |
| G6 | 最大家族 ≤ 349 | **0.500%**（v1 是 17.90%） |
| G7 | 第三方转换 ≤ 15%（全量化 ≤ 20%） | **14.999% + 2.585%**（v1 是 61.98%） |
| G8 | 无主任务 > 18%（目标 15%） | **18.000%**（v1 是 50.77%） |
| G10 | 英语侧 ≥ 可归属总体的 75% | **75.001%** |
| G16 | 元数据质量通过率 ≥ 95% | **100.000%**（v1 是 70.45%） |
| G18 | 选出数 = 可行最大值 | 69,817 = 69,817 = requested |
| G19 | 确定性复跑 id 相同 | **逐条相同** |

**一句话对比**：有效发布者数 **15.1 → 2,889.1**，有效家族数 **31.1 → 3,197.7**，
任务分布归一化熵 **0.601 → 0.918**。完整三方表见 [`T2_runs/distribution_report_v2.md`](T2_runs/distribution_report_v2.md)。

`verify_raw --forecast` 段不是闸门，是**把 T3/T5/T8 能提前算的量现在就写死**，防止事后重新叙述。
🔴 **注意：§6.3 / §8.1 / §11.3 的预测行是基于 v1 的 69,817 前缀算的，v2 换总体后必须重算**（见各节标注）。

---

## §6 T3：canonical 化与 100K 梯子构建 ✅ **已完成（2026-08-03，本地 CPU）**

> **执行记录：[`T3.md`](T3.md)。** 梯子 100,000 行已落盘，五条出闸门全过，
> `ladder_sha256 = ccf288aee2a3caee…`。**T3 不需要 GPU。**

### 6.1 `scale1m/hf_canonicalize.py`（已实现）

对每条 HF 记录产出建图所需的**四个量**（严格遵守铁律 2 —— 只要这四个）：

| 量 | 来源 | 规则 |
|---|---|---|
| `unique_model_id` | `id` | 规范化：`strip().lower()`，用于与 CORE 去重（与 `scale1m.verify_raw.normalize` **同一个函数**） |
| `size_b`（十亿参数） | 🔴 **只用 `safetensors.total`**（D-26） | 缺 ⇒ `NaN` ⇒ `param_count_to_size_bucket(None)` = bucket 0（unknown） |
| `family` | 🔴 **四级解析，目标是 CORE 的命名空间**（D-36，见下） | 不在 vocab 且计数 < `FAMILY_MIN_COUNT` ⇒ `Other`(id 0) |
| `lineage_base` | 🔴 **`baseModels` 优先**，`cardData.base_model` 兜底（D-37） | 规范化后备用；`relation` 一并保留 |

> 🔴 **F-T3-1 / D-36：原文的「复用 `attributes.py` 的 family 推断（名字前缀 + tags 兜底）」有两处错，第二处致命。**
>
> ① 真正的函数是 `dataset_embed/utils/fetch_metadata.py::_infer_one_family`，**只看名字，没有 tags 兜底**。
> ② **它和 CORE 不在同一个命名空间。** CORE 的家族串来自 ModelLens 的 `model_profile.family`，
> 实测是 `bert`(6,909) / `llama`(1,492) / `vit`(1,308) / `xlm`(1,290) / `qwen`(925) —— **HF `config.model_type` 的取值**；
> 而规则表返回 `BERT` / `LLaMA` / `ViT` / `Qwen`。`family_vocab`（341 行）里**两套都在**
>（`KNOWN_FAMILIES` 种子化在前，ModelLens 的小写串动态录在后）。
> ⇒ **HALO 的每个 llama 拿到 `LLaMA` 行，CORE 的每个 llama 待在 `llama` 行 —— 同架构两行嵌入、零共享、且不报错。**
>
> | 策略 | 落进 CORE 真正用过的 207 个家族 |
> |---|---|
> | 计划原文：`_infer_one_family` 原样 | 🔴 **0.64%** |
> | 仅小写化 | 33.98% |
> | 仅 `config.model_type` | 29.63% |
> | ✅ **四级解析（采用）** | **48.47%** |
>
> **四级解析**：① `config.model_type` 且 CORE 用过 → 用它；② 小写化的规则输出且 CORE 用过 → 用它；
> ③ `config.model_type` 非空 → 用它（新行，但命名空间对）；④ 小写化的规则输出（最后手段）。
> 每行记 `family_source`。

> 🔴 **F-T3-4 / D-37：`lineage_base` 只用 `cardData.base_model` 会丢掉 94.9% 的血缘声明。**
> 537K 候选实测：`baseModels`（HF 结构化，带 `relation`）命中 **234,445**，
> `cardData.base_model`（作者手打字符串）只有 **12,713**。改为权威优先，
> 并保留 `relation`（quantized/adapter/finetune/merge）—— 正是 CLAUDE.md 里 `r_mm'` 离散有序权重需要的量，
> **T5 不必再从名字猜**。

> 🔴 **F-T2-3 / D-26：`size_b` 不许加名字正则兜底。**
> `model_descriptor` 在 `size_b=NaN` 时**整段省略** `"<x>B params"` 子句，所以"多大比例带 size 子句"
> 本身就是一条系统性文本差异 —— **与用哪个描述符函数无关**，铁律 2 管不到它。实测：
>
> | 群 | 带 size 子句 | 与 CORE 的差 |
> |---|---|---|
> | **CORE**（`size_bucket_id ≠ 0`） | **47.91%** | — |
> | HALO 前 69,817，仅 `safetensors.total` | **38.17%** | **−9.74 pp** ✅ 选它 |
> | HALO 前 69,817，加名字正则（`-7b-`/`-350M-`） | **78.60%** | **+30.69 pp** ❌ |
>
> 名字正则能从 97,494 条缺失里救回 62,813 条（64.4%，主要是没有 safetensors 权重的 GGUF 量化仓库），
> **但救得越多离 CORE 越远。** D-18 的原则「信息量对齐优先于信息量最大化」在这里落成数字。
> 副作用照例明写：我们主动放弃了 62,813 条可从名字恢复的参数量。
>
> **更一般的教训**：铁律 2 的「同一个描述符函数」是必要条件不是充分条件 —— 同一个函数喂不同覆盖率的字段，
> 文本分布照样系统性可分。**G-B4 的 AUC 是这一层唯一的守门员。**

同时**分层打标**（进 `layer` 列，仅审计用，不进训练 —— D-10）：

| layer | 判据 | v1 前缀预测（作废） | **T3 实测（v2 HALO）** |
|---|---|---|---|
| `labeled` | 有可解析的 `model-index` 结果 | 1,642（2.35%） | **6,619（9.48%）** ×4.0 |
| `lineage` | 有 `base_model`，无 model-index | 42,109（60.31%） | **25,268（36.19%）** ×0.60 |
| `plain` | 都没有 | 26,066（37.33%） | **37,930（54.33%）** ×1.46 |
| `dropped` | 无 pipeline_tag ∧ 无 tags ∧ 无参数量（信息量为零） | 0 | 🔴 **0** —— 恒空，D-27 在 v2 上再次确认 |

> **F-T3-2：两个方向相反的力，正是 T2 回填时预告过的那一对。**
> `labeled` 涨 4 倍 —— 质量门偏好有结构化元数据的模型（⇒ **§11.1 的 `displacement_quality` 样本从 1,642 变 6,619**）；
> `lineage` 掉四成 —— 近重复上限主动削掉同根下的密集派生，而那正是 `base_model` 声明最密的一群。
> 🔴 **`plain` 过半（54.33%）必须写进报告**：超过一半的 HALO 既无标注也无可解析血缘，只能靠自身特征。

> 🔴 **F-T2-7 / D-27：`dropped` 判据是死规则，恒为 0。**
> 实测 `tags` 覆盖率 **100.00%** —— HF 给每个仓库自动挂 `region:us` / `license:*` / 架构名 之类的标签，
> 所以那个合取式**永远为假**。
> **定案：保留判据不动**（`layer` 只进审计不进训练，恒空不产生任何下游影响；改判据等于引入一条我们没验证过的新过滤规则，
> 风险大于收益）。**但报告里必须写明它恒为 0，不能让读者以为过滤生效了。**

```bash
python -m scale1m.hf_canonicalize --candidates $C --core .../hgraph_ml_v2.pt
# 产出 canon/hf_canon.parquet（model, id_norm, size_b, family, lineage_base, layer,
#   downloads, rank + size_source / family_source / lineage_source / family_in_core_used）
```
输入是 **`candidates_v2/`**，不是 v1 的 `raw/`（后者已按 D-30 降级为审计件）。

### 6.2 `scale1m/build_ladder.py`（已实现，主体是断言）

```python
# v2 (D-30): HALO 不再是"爬取顺序的前 69,817 条",而是已通过 G1-G20 的均衡总体
core_ids = set(normalize(m) for m in core_umi["model"])          # 30,183
halo_100k = pd.read_parquet(SELECTED_HALO)                       # 69,817, 已均衡、已过门
assert len(halo_100k) == 100_000 - len(core_ids)
assert not (set(halo_100k.id_norm) & core_ids)   # 选样阶段未排除 CORE -> 这里必须查

# 🔴 mappedID 纪律：CORE 占 0..30182 原序不动，HALO 从 30183 起
ladder = pd.concat([core_umi.assign(layer="core"), halo_100k.assign(mappedID=...)])
assert ladder.mappedID.tolist() == list(range(100_000))
assert ladder.iloc[:30183]["model"].tolist() == core_umi.sort_values("mappedID")["model"].tolist()
```

concat **之前**四项、**之后**四项，共八条断言（行序错位不会抛异常，只会悄悄改写"哪个模型拥有哪一行嵌入"）：

```python
assert len(halo) == n - n_core                     # HALO 规模
assert not (core_id_set & set(halo.id_norm))       # 🔴 CORE∩HALO=0（D-35）
assert halo.id_norm.nunique() == len(halo)         # HALO 内部无重复
assert not (set(halo.id_norm) - set(canon.index))  # canon 覆盖全部 HALO
# --- concat ---
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

**产物**：`ladder/100k_model_ids.csv`（`mappedID, model, layer, size_b, family, lineage_base`）
+ `LADDER_REPORT.json`（sha256 与全部审计统计）。

> 🔴 **D-38：梯子的 CORE 侧（0..30182）`size_b`/`family`/`lineage_base` 一律留空，`layer="core"`。**
> 铁律 1 说 CORE 这些量从冻结图**原样复制、绝不重算**，梯子不去碰；T4 直接从 `hgraph_ml_v2.pt` 读。
> 梯子在 CORE 侧只承担一件事 —— **钉住 id 与行序**。

### 6.3 T3 出闸门 —— ✅ 五条全过

- [x] `len(ladder) == 100000` 且 `mappedID` 是 `0..99999` 的连续整数
- [x] 🔴 `ladder[:30183]["model"] == CORE 的 mappedID 序`（逐项相等）
- [x] HALO 与 CORE **零交集** ⇒ **0**，湖内 distinct = **100,000**（选样阶段已 `--exclude-core`，D-35）
- [x] `layer` 分布已统计并记录（labeled 6,619 / lineage 25,268 / plain 37,930 / dropped 0）
- [x] `size_b` 缺失率 **42.25%**、`family` 统计已记录（见下）

**T2 已经把这一节的数算出来了，T3 的任务是对账，不是发现：**

| 量 | v1 前缀预测（作废） | **T3 实测（v2 HALO）** | CORE 参照 |
|---|---|---|---|
| HALO 来源 | 爬取顺序前 69,817 | **537,025 候选 → 配额选出 69,817** | — |
| `size_b` 已知（= 带 size 子句） | 38.17% | **57.75%** | CORE **47.91%**（**差 +9.83 pp**） |
| `size_b` 缺失率 | 61.83% | **42.25%** | CORE **52.09%** |
| `family` 不同取值 | — | **9,292** | 207（CORE 用到的） |
| `family == "other"`（规则层） | — | **0.23%** | — |
| **`family_id == 0` 预测**（过 `FAMILY_MIN_COUNT=3` 折叠） | 待实测 | **12.78%** | CORE **34.16%** |
| 落进 CORE 用过的家族 | — | **48.47%** | — |

⇒ **远低于下面 80% 的警戒线，也低于 CORE 自己的 34.16%，`e_fam` 不会塌成一行。**
但另一半（51.53%）会新建 vocab 行 ⇒ **T4 的 `family_vocab` 会从 341 行扩到约 2,000 行**。
这不违反铁律 1（CORE 已有行 id 绝不移位，T4 有断言），但 checkpoint 与 vocab 的绑定必须同步更新。

> 🔴 **D-26 的论证需要在 T4 按新数字改写（结论不变且更强）。**
> v2 的质量门偏好有结构化元数据的模型，`safetensors` 覆盖率**反超** CORE **+9.83 pp**（v1 时是落后 9.74 pp）——
> **偏离方向翻转了。** 原始理由"向 CORE 覆盖率靠拢"不再成立；
> 新论证是"两个方向都偏离，选偏离更小的那个"（加名字正则会推到 ~80%，偏离更远）。

> ⚠️ **若 `family=Other` 占比 > 80%**：说明 family 推断规则对 HF 长尾失效，HALO 的 `e_fam` 全塌到一行。**这不阻塞实验**（Other 本就是设计内的降级路径），但必须在报告里写明，且 C 轴的 frozen 层分析要把它列为混杂因素。
> **参照系已经有了：CORE 自己就是 34.16% Other。**"HALO 比 CORE 高多少"比"HALO 是否超过 80%"更有信息量 —— 两者都记。

> 🟠 **F-T2-5（v2 后已大幅缓解）：CORE 与 HALO 的 popularity 差异。**
> v1 时 HALO 是纯头部群体；v2 的 popularity 配额（head 32.1% / mid 23.2% / long-tail 11.1% / recent 33.6%）
> 已经把它摊开。`downloads` 不是节点特征，所以这不直接破坏铁律 2，但仍留在 G-B4 的归因清单里。

---

## §7 T4：特征构建

### 7.1 新建 `scale1m/embed_lake.py`

**分两段，严格区别对待：**

```python
# ── 段 1：CORE（30,183 行）—— 原样复制，绝不重算（铁律 1）──────────
core = torch.load(CORE_GRAPH, weights_only=False)
x_core        = core["data"]["model"].x                     # [30183, 448]
size_id_core  = core["data"]["model"].size_bucket_id
fam_id_core   = core["data"]["model"].family_id

# ── 段 2：HALO（69,817 行）—— 新算，但用 CORE 的描述符函数（铁律 2）──
from scale.modellens_build_graph import model_descriptor    # 🔴 就是这一个
texts   = [model_descriptor(mid, fam, sz) for mid, fam, sz in halo_rows]
e_name  = build_name_embeddings(halo_ids, token_dim=64, seed=42)      # 与 CORE 同 seed
e_desc  = minilm_batched(texts, batch_size=1024, fp16=True)           # all-MiniLM-L6-v2
x_halo  = np.concatenate([e_name, e_desc], axis=1)                    # [69817, 448]

x = torch.cat([x_core, torch.from_numpy(x_halo)], dim=0)              # [100000, 448]
```

**size/family id：**
- `size_bucket_id`：HALO 走 `param_count_to_size_bucket(size_b * 1e9)`，与 CORE 同一套固定常数
- `family_id`：🔴 **`load_or_update_family_vocab` 是 append-only** —— 加载 CORE 的 `family_vocab.csv`，HALO 的新家族只在计数 ≥ `FAMILY_MIN_COUNT` 时**追加新行**，CORE 已有行的 id **绝不移位**

```python
vocab_before = dict(core["xm0_meta"]["family_vocab"])
vocab_after  = load_or_update_family_vocab(all_families, vocab_path=FAMILY_VOCAB)
for k, v in vocab_before.items():
    assert vocab_after[k] == v, f"family row moved: {k} {v} -> {vocab_after[k]}"   # 🔴
```

### 7.2 执行（GPU 作业）

```bash
sbatch --gres=gpu:1 --mem=32G --time=1:00:00 --wrap \
 "python -m scale1m.embed_lake --rung 100k --out \$WORK/model_lake/data1m/feats"
```
**预估：69,817 条 MiniLM，A100 fp16 batch 1024 ⇒ 2–5 分钟。**（1M 时 10–25 分钟）

### 7.3 T4 出闸门

- [ ] 🔴 `torch.equal(x[:30183], x_core)` —— CORE 特征逐字节未变
- [ ] `x.shape == (100000, 448)`，无 NaN
- [ ] `family_vocab` 只增不移（上面的断言）
- [ ] 行序 checksum：随机抽 100 个 HALO 行，按 `ladder` 里的 id 重算 `e_name`，与 `x[i,:64]` 比对相等
- [ ] `x[30183:]` 与 `x[:30183]` 的**文本长度分布**统计对比 —— **铁律 2 的实证检查**：两群的 `len(descriptor)` 分布应大致重叠。若 HALO 明显更短/更长，回查描述符是否用错

**T3 已经把名字那一半（`e_name` 的输入）在 v2 总体上量过了，T4 只需补 `e_desc` 那一半：**

| | 字符 mean | 带组织前缀 | 带 size 子句 |
|---|---|---|---|
| CORE | **38.9** | 80.6% | 47.91% |
| HALO v1 前缀（作废） | 39.2（几乎重合） | 100.0% | 38.17% |
| **HALO v2 实测** | **33.8**（🔴 **短 5.1 字符**） | 100.0% | **57.75%** |

> 🔴 **F-T3-5/F-T3-7：v1 时"名字长度分布对齐得很好"这条好消息在 v2 没有了。**
> 均衡选样削掉了名字冗长的量化搬运仓库（`X-i1-GGUF`、`X-Q4_K_M-GGUF`），HALO 名字平均短了 5.1 个字符。

**AUC 偏高时按此顺序归因（v2 更新）：**

1. 🔴 **size 子句覆盖率差 +9.83 pp**（F-T3-5）—— 最大的一条，且是设计内的（D-26）。**注意符号翻转了**：v1 时 HALO 更少，v2 时 HALO 更多
2. 🔴 **名字平均短 5.1 字符**（F-T3-7，v2 新增通道）
3. **CORE 有 19.4% 的裸名字**（`bert-base-uncased`、`#-shots` 这类），HALO **一个都没有**
4. **HALO 组织集中度已不成问题**：前五名 `WindstormLabs` 0.90% / `mradermacher` 0.50% / `OpenMed` 0.50% / `Helsinki-NLP` 0.41% / `mlx-community` 0.35%（v1 时是 `mradermacher` 一家 20.1%）
5. popularity 差异（F-T2-5，v2 后已大幅缓解）

> 💡 **一个额外的诊断（强烈建议做，10 行代码）**：训一个逻辑回归，只用 `x` 去分类「这一行是 CORE 还是 HALO」。**若 AUC > 0.9，铁律 2 就已经被破坏了** —— 两群在特征空间里可被平凡分开，后面的 gold@10 不降就是假的。AUC 应该在 0.5–0.75 之间（有真实分布差异是正常的，可平凡分离不正常）。
> **AUC 偏高时不要立刻回滚描述符** —— 先按上面四条归因。第 2/3/4 条是湖的真实构成，不是 bug；只有第 1 条是我们能调的旋钮，而它已经调到最接近 CORE 的一档了。

---

## §8 T5：建图

### 8.1 新建 `scale1m/build_graph_rung.py`（改自 `scale/modellens_build_graph.py`）

| 边类型 | 100K 上怎么建 | 预估量 |
|---|---|---|
| `trained_on` + rev | 🔴 **从 CORE 原样复制**（HALO 无监督边） | 312,986（与 CORE 完全相同） |
| `similar_to` (d–d) | 🔴 **从 CORE 原样复制**（数据集节点完全不动） | ~192,060 |
| `is_base_of` + rev | HALO 的 `lineage_base` 对 100K 的 id 表做 **hashmap join**（不是 O(N²) 名字匹配） | 🟢 **预测 26,297**（T2 实测算出） |

```python
# 血缘边：规范化 id 的 hashmap join，O(N)
id2idx = {normalize(m): i for i, m in enumerate(ladder["model"])}
src, dst = [], []
for i, base in zip(ladder.index, ladder["lineage_base"]):
    j = id2idx.get(normalize(base))
    if j is not None and j != i:
        src.append(j); dst.append(i)          # base --is_base_of--> derivative
```

**必须产出 `lineage_stats.json`**（无论结论好坏 —— `plan.md` §3.2 明文要求）：
```json
{"total_edges":?, "core_core":?, "core_halo":?, "halo_halo":?,
 "n_models_with_lineage":?, "n_components":?, "largest_component":?,
 "halo_lineage_coverage": ?}
```

> 🟢 **D-8 的第一次真实检验 —— T3 已用 v2 总体重算（`T3.md` §4.2）**：
>
> | 量 | P1 名字匹配 | v1 前缀预测（作废） | **T3 实测预测（v2）** |
> |---|---|---|---|
> | HALO 声明 `base_model` | — | 42,937（61.50%） | **27,672（39.64%）** |
> | 其中在 100K 湖内可解析 ⇒ **`is_base_of` 边数** | **42** | 26,297 | **16,385** |
> | ├ `core_halo` | — | 560 | **606** |
> | ├ `halo_halo` | — | 25,737 | **15,779** |
> | └ `core_core` | — | 0 | **0** |
>
> **仍是 P1 的 ≈390×，但比 v1 口径少 38%。** 原因是近重复上限削掉的正是血缘最密的一群
>（同一 base 的 20 个 GGUF 变体各连一条）。**这是取舍不是退步** —— 那些边对 hub 过平滑是负担而非信息。
> **T5 实测时必须把这两个数并列报，不许只报有利的那个。**
>
> 🔴 **`relation` 已经从 `baseModels` 拿到（D-37）** —— quantized / adapter / finetune / merge 是 HF 的结构化字段，
> CLAUDE.md 里 `r_mm'` 的离散有序权重**不必再从名字猜**。
>
> 下面是 v1 口径的旧预测，保留供对账：
>
> | 量 | 预测值 |
> |---|---|
> | HALO 前 69,817 中声明 `base_model` 的 | **42,937（61.50%）** |
> | 其中 base 能在 100K 湖内解析到的 ⇒ **`is_base_of` 边数** | **26,297**（占声明数 61.25%） |
> | ├ `core_halo` | **560** |
> | ├ `halo_halo` | **25,737** |
> | └ `core_core` | **0** |
> | P1 名字精确匹配的老基线 | 42 |
>
> **≈626×。** T5 的任务是**对账**，不是重新发现；实测若显著偏离这三个数，说明 `normalize()` 或 join 有 bug。
>
> 🔴 **但这个好消息带一个必须一起说的限制**：`core_core = 0`、`core_halo` 只有 560
> ⇒ **血缘结构几乎全部长在 HALO 内部，而 gold 标签只存在于 CORE。**
> ⇒ §11.3 C 轴的问题 3（「`cold` 是否显著优于 `frozen`」）**不能用 `gold@10` 回答**（26,297 个 cold 节点里几乎没有 gold 候选），
> 只能用表示质量类指标回答。已回填 §11.3。
>
> `core_core = 0` 的成因与一个**明确的不作为**：CORE 的 30,183 行来自冻结的 `hgraph_ml_v2.pt`，本来就没有 HF `cardData`。
> 我们**爬到了** 1,199 个 CORE 模型的 HF 记录（含其 `base_model`），**但不会用它给 CORE 子图加边** ——
> 加了就等于改动 CORE，R0/R1/R2 之间「N 是唯一变量」当场破功。这是纪律，不是遗漏。

### 8.2 执行

```bash
python -m scale1m.build_graph_rung --rung 100k \
  --core   $WORK/model_lake/data1m/graphs/hgraph_ml_v2.pt \
  --ladder $WORK/model_lake/data1m/ladder/100k_model_ids.csv \
  --feats  $WORK/model_lake/data1m/feats/100k \
  --out    $WORK/model_lake/data1m/graphs/hgraph_100k.pt
```
**预估**：文件 ~250 MB，构建 3–10 分钟。

### 8.3 T5 出闸门（= §12 的 G-B）

```python
# scale1m/verify_rung_graph.py --rung 100k  —— 一条命令跑完全部
assert data["model"].num_nodes == 100_000
assert data["dataset"].num_nodes == 9_603
assert data["model"].x.shape == (100_000, 448)
assert data["dataset"].x.shape == (9_603, 458)
assert len(data.edge_types) == 5
assert not torch.isnan(data["model"].x).any()
# 铁律 1 的六条（见 §1）
# xm0_meta / xd0_meta 与 CORE 逐键一致（family_vocab 只允许变长）
```

- [ ] `verify_rung_graph --rung 100k` 全绿
- [ ] `stage1BuildTransferGraph/check_stage2_contract.py` 通过
- [ ] `lineage_stats.json` 已产出
- [ ] 🔴 **划分不变性**：用 `make_root_aware_splits(data, roots, split_seed=0)` 在 100K 图与 CORE 图上各跑一次，**test 边集合的 sha256 必须相同**（HALO 无监督边 ⇒ 划分只由 CORE 决定 ⇒ 必然相同；不同就是有 bug）

---

## §9 T6：训练

### 9.1 新建 `scale1m/train_rung.py`（改自 `scale/export_ours.py`）

复用 `export_ours` 的全部逻辑，增加：`--expect-n`、`--chunked-infer`、`--sparse-M`、`--loader fanout`、`--amp`、`--resume`、`--out`。

配置**零改动**沿用 `l1l3b_config()`（[export_ours.py:50](../../scale/export_ours.py#L50)）：L1 全湖 logQ 采样 softmax + L3 native task + `global_n_neg=256` + `batch_size=1024`。

### 9.2 三步走（不许跳）

```bash
# ── 步 1：R0 锚点（12K）在 A100 上复现 ──────────────────────
RUNG=12k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
#    ⇒ 必须 gold@10 = 0.4159 ± 3-seed 噪声。不过 → 停，查环境/AMP/loader

# ── 步 2：R1（30K 全 CORE，D-9 解除）───────────────────────
RUNG=30k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh
#    ⇒ 这是 10 个月来第一次跑通全 CORE，本身是独立交付

# ── 步 3：R2（100K）先冒烟再全量 ────────────────────────────
RUNG=100k SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh    # 冒烟
#    ⇒ loss 下降、无 NaN、显存/墙钟记录 ⇒ 据此定 --time
RUNG=100k SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh    # 全量（seed 0/1/2 各一次）
```

**预估单次（A100，25 epochs）**：R0 ~5 min / R1 ~15 min / R2 ~30 min。
监督边数三档不变（312,986），代价增量来自采样与推理，**不是来自监督量** —— 这本身就是「监督冻结」设计的一个好处。

### 9.3 checkpoint（CLAUDE.md Step 6 的硬要求）

每 5 epoch 及验证边界保存 checkpoint。内容包括 model、optimizer、scheduler、AMP gradient scaler、epoch、global step、best metric/early-stopping 状态、Python/NumPy/CPU torch/CUDA torch 随机状态、resolved config 与数据版本；可行时同时保存 sampler/DataLoader 进度。模型状态必须包含 encoder 的两张 embedding 表，并记录 `family_vocab.csv` 路径与 SHA-256、size bucket 常数版本、`e_name` seed、`token_dim` 与 encoder 名。

checkpoint 先写临时文件，完成 flush/close 后原子 rename 为 `checkpoint_step_<N>.pt`；`last.pt`/`latest` 仅在目标文件成功后更新。运行目录保留 `best`、`latest` 与至少一个历史 checkpoint，禁止覆盖唯一有效副本。

resume 解析顺序固定：

1. 显式 `--resume <path>`；
2. `$OUT/ckpt/last.pt` 或 `latest`；
3. 新运行。

日志必须记录最终选择的 checkpoint。恢复前校验模型结构、resolved config、数据/划分标识、`family_vocab` 与 size bucket 版本；不兼容时明确失败。Slurm preemption/requeue 继续使用同一 logical `RUN_ID`，恢复 optimizer/scheduler/scaler 与 step 后再推进，避免重复 metrics。完整训练提交前必须通过 save、reload、resume 后 step 连续递增的测试。

### 9.4 T6 出闸门

- [x] G1 机制：loss 单调下降、无 NaN、两张 embedding 表梯度非零、frozen `x` 梯度为 `None`（2026-08-11，六个 T6 run 的机制门全过）
- [ ] 🔴 G6 锚点：`gold@10(R0) = 0.4159 ± 噪声`
- [x] R1 训练成功（**D-9 正式解除**；30K/25 epoch，loss 18.9898 → 15.0575）
- [x] `MANIFEST.json` 已写（含 `peak_gpu_mem_gb` / `wallclock_s`；本地 evidence 合约复验通过）

---

## §10 T7：导出与索引

```bash
python -m scale1m.export_rung --rung 100k --run $OUT \
  --chunk 50000 --hnsw-M 32 --ef-construction 200 --hnsw-threads 8
```

**产物：**

| 文件 | 内容 | 检查 |
|---|---|---|
| `z_m.npy` | `[100000,128]` 全图前向（服务用） | 行序断言 |
| `z_m_eval.npy` / `z_d_eval.npy` | **held-out 前向**（🔴 A 轴数字**只能**用这个） | 见下 |
| `hnsw_100k.bin` | HNSW 索引 | recall ≥ 0.99 |
| `model_ids.csv` | `mappedID → unique_model_id` 快照 | 与 ladder 逐行相等 |
| `gold_cands.npz` | held-out gold 标签 | 与 CORE 相同 |
| `ef_tuning.json` | iso-recall 二分过程 | — |

**🔴 泄漏门（G-D2）**：P4 抓到过这个坑 —— 全图 `z_d` 让 held-out 查询看到自己的监督边，`gold@10` 从 0.42 虚高到 0.61。
```python
assert gold10(z_eval) < gold10(z_full), "用反了：A 轴必须用 held-out 前向的 z_*_eval"
```

**🔴 行序门（G-D3）**：随机抽 100 个 `mappedID`，反查 `model_ids.csv` 的 `unique_model_id`，与 `ladder` 逐项相等。

---

## §11 T8：三轴评测

### 11.1 A 轴：精度

```bash
python -m scale1m.eval_rung --rung 100k --run $OUT --expect-n 100000
```

主表（R0/R1/R2 三行，R3/R4 后续补）：

| N | gold@1 | **gold@10** | top3@10 | gold-gap@10 | root_gold@10 | median gold rank | **median rank / N** | vs random |
|---|---|---|---|---|---|---|---|---|
| 12,000 | 0.0948 | **0.4159** | 0.5319 | 0.5087 | 0.3920 | 21 | 1.75×10⁻³ | 4.99×10² |
| 30,183 | | | | | | | | |
| 100,000 | | | | | | | | |

- `vs random` = `gold@10 / (10/N)` —— 100K 时随机基线是 **1×10⁻⁴**
- 🔴 **`median rank / N` 是核心列**，不是 `gold@10`。它衡量「gold 在池子里的相对位置」，是唯一跨 N 可比的量。

**📌 预注册预测（先写下来，防止事后编故事）：**

> 若表示的判别力**与规模无关**（ranks 随 N 按比例放大），则 100K 的 `gold@10` ≈ 12K 的 `gold@1.2` ≈ **0.10–0.12**。
> 若表示对 HALO 这类**无标签长尾**有额外判别力（大概率有：它们多是离题模型，容易被推远），则会**高于**这个带，落在 **0.15–0.30**。
>
> - 落在 **0.10–0.30** ⇒ 符合预期，正常报告
> - **> 0.35**（几乎没掉）⇒ 🔴 **先查 bug，不是先庆祝**。查三处：① 候选池真的是 100,000 吗（铁律 3）② HALO 的描述符用对了吗（铁律 2 + §7.3 的 AUC 诊断）③ 泄漏门 G-D2
> - **< 0.05** ⇒ 表示在放大后失效，是真结果，照实报，并做 C 轴的分层归因

**新增 `displacement_quality`**（D-10 的免费副产品）：
```
对每个 held-out 查询，取「排在 gold 前面的 HALO 模型」；
其中 layer == "labeled" 的那些，它们在该数据集上的实测精度 vs gold 的精度：
  - 显著更低 ⇒ 真错误
  - 接近/更高 ⇒ 挤占其实发现了 CORE 语料没标注的好模型，gold@10 低估了真实效用
两种情形都照实报，占比写清楚。
```

> 🟢 **F-T3-2 更新：样本基数从 1,642 涨到 6,619（×4.0），这个分析可以给定量结论了。**
> v1 口径下 HALO 只有 1,642 条 `layer=labeled`（2.35%）；v2 均衡选样的质量门偏好有结构化元数据的模型，
> 实测 **6,619 条（9.48%）**。
> 仍然要连分母一起报（"N 个挤占者中 M 个 labeled"），但已经不必降级成纯定性叙述。

### 11.2 B 轴：iso-recall 延迟

**🔴 协议（比数字重要，不许简化）：**

1. **iso-recall**：`tune_ef_for_recall(target=0.99)` 二分出最小 `ef_search`，**然后**才计时
   > 不这么做，「你只是把 recall 调低换速度」一句就打穿
2. **单线程**：`idx.set_num_threads(1)`，逐查询、禁 batch
3. **warmup 100 次丢弃**，再测 1000 次
4. 报 **p50 / p95 / p99**，不只报均值
5. **同一次作业内跑完 R0/R1/R2 三档**（跨节点 CPU 差异会污染曲线）
6. 另报多线程 **QPS**（部署关心的是这个）

| N | ef_search | recall@50 | p50 (ms) | p95 | p99 | 建索引 (s) | 索引内存 (MB) | ModelLens Θ(N) **外推** p50 | 加速比 |
|---|---|---|---|---|---|---|---|---|---|
| 12,000 | | ≥0.99 | | | | | | | |
| 30,183 | | ≥0.99 | | | | | | | |
| 100,000 | | ≥0.99 | | | | | | | |

🔴 ModelLens 列**必须标注「外推，非实测」**（D-16）。外推法写清：用 P4/P5 在 1K–12K 实测的点拟合 Θ(N) 直线并给区间。

**曲线拟合**（三点还太少，R3/R4 后才裁决，但先算）：对 p50 拟合 `a·log N + b` 与 `a·N^α + b`，报 α 与置信区间。

### 11.3 C 轴：冷启动分层

| 层 | 定义 | 12K 占比 | **100K 预测**（T2 算出） | 100K 实测 | gold@10 | z_m 层内平均余弦 | 有效维度 |
|---|---|---|---|---|---|---|---|
| warm | `trained_on` 度 ≥ 10 | | 5,765（**5.76%**）确定值 | | | | |
| cool | 度 1–9 | | 24,369（**24.37%**）确定值 | | | | |
| cold | 度 = 0 **有血缘** | | ~~26,297~~ **16,385（16.39%）** | | | | |
| **frozen** | 度 = 0 **无血缘** | ~0 | ~~~65%+ / 43,569~~ **53,481（53.48%）** | | | | |

> ✅ **已按 v2 总体重算（`T3.md` §4.6）。** `warm`/`cool` 完全来自 CORE，是确定值；
> `cold`/`frozen` 随 §8.1 的血缘边数一起改写。
>
> 🔴 **`frozen` 从 43.6% 涨到 53.48%，成为最大的一层。**
> C 轴的重心随之移动 —— "`frozen` 层能否被检索到、是否塌缩"从一个次要问题变成**过半总体的问题**。
>
> **F-T2-11（v1 口径，已作废）：`frozen` 43.6%。**
> 原数字是在「血缘边很少」的假设下写的；F-T2-4 把 26,297 条边算出来后，四分之一强的 HALO 从 `frozen` 移进了 `cold`。
> `warm`/`cool` 完全来自 CORE（HALO 无监督边），所以这两行是**确定值**不是预测。

**要回答：**
1. `frozen` 层能被检索到吗？（它们的 `z_m` 完全来自 `[e_name‖e_desc‖e_size‖e_fam]` + 空邻域 —— 正是 edge-dropout 训练的降级路径）
2. `frozen` 层是否塌缩？测层内平均余弦、有效维度、与 warm 层的分布距离。**全塌到一个点 = edge-dropout 的降级路径没学好，是必须诚实报告的负面结果**
3. 🔴 `cold`（有血缘无标签）是否显著优于 `frozen`？**这一问必须换指标**：
   F-T2-4 测出血缘几乎全在 HALO 内部（`core_core=0`、`core_halo` 仅 560），而 **gold 标签只存在于 CORE**
   ⇒ 26,297 个 `cold` 节点里几乎没有一个是 gold 候选 ⇒ **`gold@10` 在 cold/frozen 之间没有对比力，用它回答会得到"两边都约等于 0"的假平局。**
   改用**表示质量**回答：层内平均余弦、有效维度、与 warm 层的分布距离、以及
   「同一个 base 的派生模型之间的余弦 vs 随机对的余弦」（血缘边是否真的把派生族拉近了，同时**没有**拉塌）。
   **若 `cold` 在这些量上显著优于 `frozen`，这就是 `is_base_of` 这条 novel contribution 的第一份规模化证据** —— 但要说清楚它是表示层证据，不是检索精度证据。

### 11.4 D 轴：工程

| 指标 | 12K | 30K | 100K |
|---|---|---|---|
| 端到端建库墙钟（爬取→嵌入→建图→训练→索引） | | | |
| 峰值 GPU / CPU 内存 | | | |
| 索引磁盘 | | | |
| **增量上线一个新模型**（特征→前向→`add_items`） | | | |
| 多线程 QPS | | | |

最后一行是产品级论据：**新模型上线 O(1)，不重训、不重建索引** —— ModelLens 的 id-embedding 架构做不到（新模型没有 id 行）。

---

## §12 gate 总表（逐条勾）

| ID | 阶段 | 判据 | 不过怎么办 |
|---|---|---|---|
| **G-W1** | T1 | SSH access、quota、大文件目录、partition/account 均有实测记录 | 暂停上传与提交，向管理员确认 |
| **G-W2** | T1 | 环境依赖通过，CUDA 在 allocation 内可见，tiny batch forward/backward 通过 | 修复环境或资源申请后重跑 smoke test |
| **G-W3** | T1 | Git commit、数据 manifest/checksum 已记录，resolved config 无 Windows 路径 | 修正集中配置并重新同步/校验 |
| **G-W4** | T1/T6 | checkpoint save/reload/resume 后 step 连续，状态完整恢复 | 禁止提交 25-epoch 作业，先修复恢复逻辑 |
| **G-W5** | T1 | tiny `sbatch` 完成，log、metadata、exit code 与 job ID 正确落盘 | 修正 Slurm wrapper 与输出目录 |
| **G-W6** | T8 | checkpoint、metrics、分片、commit/config 与最终 Slurm 状态一致 | 运行保持未验收状态并补齐审计 |
| **G-A1** | T0 | 12K 锚点 `gold@10` 3-seed CI 与基线重叠 | 逐项回滚改造定位 |
| **G-A2** | T0 | 分块推理 vs 全图 `max\|Δ\| < 1e-5` | 查 `n_id` scatter |
| **G-A3** | T0 | `five_metric_eval == global_metrics` 逐查询 | 查流式打分改写 |
| **G-A4** | T0 | D-14 A/B 对拍表已产出 | 必须做，无论结论 |
| **G-B1** | T5 | 铁律 1 六条断言全过 | 🔴 停 |
| **G-B2** | T5 | `check_stage2_contract` 通过 | 🔴 停 |
| **G-B3** | T5 | 划分 test 边 sha256 与 CORE 相同 | 🔴 停，有 bug |
| **G-T2a** | T2 | ✅ v1 快照门：150,000 条 / 3 片 sha256 复算一致 / 只读 / 快照日 2026-08-03 / 0 重复 | — |
| **G-T2b** | T2 | ✅ v2 总体门 G1–G20（`verify_raw --selected`），**全部按实际选出的 69,817 计算**：18/20 硬门过，G14/G15 为联合约束所限并已定量披露 | 🔴 停，不许把未过门的总体叫 frozen |
| **G-T2c** | T2/T3 | ✅ 已过：**`CORE ∩ HALO = 0` 且 `|CORE ∪ HALO| = 100,000`** —— G1–G20 查不到这一条（G1 只查 HALO 内部重复），由 `build_ladder` 断言 | 🔴 停；用 `--exclude-core` 重跑选样（D-35） |
| **G-T3** | T3 | ✅ 已过：梯子 100,000 行 / `mappedID` 连续 / **CORE 前缀逐项相等** / 全表 id 去重 = 100,000；`ladder_sha256` 已记 | 🔴 停，行序错位不会自己暴露 |
| **G-B4** | T4 | CORE/HALO 可分性 AUC ∈ [0.5, 0.75] | > 0.9 ⇒ **先按 §7.3 的四条通道归因，再考虑回查描述符**（size 覆盖 −9.7 pp / CORE 19.4% 裸名字 / HALO 前五组织占 31.2% / popularity 差异，都是湖的真实构成而非 bug） |
| **G-C1** | T6 | 机制门（loss↓ / 无 NaN / 梯度边界） | 回 T0 |
| **G-C2** | T6 | R0 锚点 = 0.4159 ± 噪声 | 🔴 停 |
| **G-D1** | T8 | 🔴 候选池 `z_m.shape[0] == 100000` | 🔴 停，铁律 3 |
| **G-D2** | T8 | 泄漏门 `gold10(z_eval) < gold10(z_full)` | 🔴 停 |
| **G-D3** | T8 | 行序门（抽 100 反查） | 🔴 停 |
| **G-D4** | T8 | HNSW `recall@50 ≥ 0.99` | 调 ef/M；调不上去查过平滑 |
| **G-D5** | T8 | **单调性** `gold@10(100K) ≤ gold@10(30K) ≤ gold@10(12K)` | 🔴 上升 = 泄漏或采样偏置，**不许当好消息报** |
| **G-D6** | T8 | 预注册区间 `gold@10(100K) ∈ [0.10, 0.30]` | 超出上界先查 bug（见 §11.1） |

---

## §13 失败排查手册

| 症状 | 最可能原因 | 先查哪 |
|---|---|---|
| 作业长期 `PENDING` | partition/account 无权限，或 GPU/CPU/RAM/constraint 请求过严 | `squeue -t PENDING` 与 `scontrol show job -dd`；根据 pending reason 修正请求，避免反复取消重投 |
| 环境/import 失败 | batch shell 未激活正确环境，或依赖与节点 CUDA 不兼容 | `which python`、`python -m pip check`、环境列表与 Slurm 激活段 |
| 远端出现 `C:\\` / `D:\\` | local-only path 进入代码或 resolved config | 搜索仓库与配置，改为 `PROJECT_ROOT`/`DATA_ROOT`/`OUTPUT_ROOT` |
| 作业被 preempt/requeue | checkpoint 或 logical `RUN_ID` 恢复不完整 | 检查 resume 日志、step 连续性、optimizer/scheduler/scaler 与 metrics 去重 |
| `gold@10` 几乎没掉（>0.35） | ① 候选池仍是 30K ② HALO 描述符用错 ③ 泄漏 | G-D1 → G-B4 → G-D2 |
| `gold@10` 塌到 ~0 | HALO 特征 NaN / 全零；或 `x[:30183]` 被覆盖 | G-B1；查 `x[30183:]` 的 norm 分布 |
| CUDA OOM（训练） | 扇出上限没生效，或对比损失仍全 N² | 先记录失败配置与峰值；修正采样，或减小 per-device batch 并用 gradient accumulation 保持 effective batch；学习率变化须显式登记 |
| CUDA OOM（导出） | 分块推理没走上 | 确认 `--chunked-infer` 生效 |
| CPU OOM | 稀疏 `M` 没生效；或 `scores_by_q` 仍物化 | 查 `M.is_sparse` 与流式打分；依据 `MaxRSS` 调整 worker/prefetch 或 `--mem` |
| HNSW recall 上不去 | hub 附近过平滑（CLAUDE.md 三大风险之三） | 先调 `ef_search`↑；仍不行 ⇒ 加强同 hub 负样本或减一层。**100K 上第一嫌疑是量化搬运仓库**：`mradermacher` 独占 HALO 的 20.1%（14,061 个），它们既共享 `e_name` 的组织 token、又通过 `base_model` 指回各自原模型 —— 过平滑在 100K 上会比 12K 严重得多（F-T2-10） |
| 血缘边 ≪ 26,297 | `base_model` 规范化不匹配（大小写 / 前缀），或 join 写成了对 CORE-only 的表 | 与 `T2.md` F-T2-4 的三个预测值（26,297 / core_halo 560 / halo_halo 25,737）逐项对账；抽 20 条人工比对 `normalize()` |
| `size_b` 100% 缺失 | 🔴 爬取走了 `full=true` 而不是 `expand[]`（F-T2-1） | 查 `PROVENANCE.json` 的 `expand` 字段里有没有 `safetensors`；有则查 T3 的取值路径 |
| 划分 sha256 不同 | CORE 的 `trained_on` 边序被改了 | 查 G-B1 的 `edge_index` 断言 |
| 作业撞 walltime | 单 epoch 估错或未按期 checkpoint | 从有效 checkpoint 续跑；按已完成 step 实测申请时限，或拆为可恢复阶段 |
| `family_vocab` 断言炸 | `load_or_update_family_vocab` 不是 append-only 调用 | 确认传了 CORE 的 `vocab_path` |

---

## §14 产物目录

```
$WORK/model_lake/          （本地为 %MLF_DATA_DIR% = D:\research\model_lake\data）
├── data1m/
│   ├── raw/           ✅ v1 冻结件：hf_models_{00000..00002}.jsonl.gz（150,000 条 / 24.8 MB / 只读）
│   │                     PROVENANCE.json（只读）  SHARDS.json  CURSOR.json
│   │                     —— 审计件与候选源，**不再是 HALO 的定义**（D-30）
│   ├── v1_audit/      ✅ annotated_v1.parquet（"before" 对照）
│   ├── candidates_v2/ ✅ hf_models_000{00..11}.jsonl.gz（537,025 条 / 64.7 MB / 只读）
│   │                     membership/*.tsv.gz（532 个查询各自的命中列表 = 溯源合并依据）
│   │                     PROVENANCE.json  PLAN_STATE.json  SHARDS.json
│   │                     annotated.parquet  ANNOTATE_REPORT.json  AUTHOR_STATS.json
│   │                     selected/       selected_halo.parquet/.csv（🔴 T3 的 HALO 输入）
│   │                                     SELECTION_MANIFEST.json  DEFICITS.json  EXIT_GATES.json
│   │                     selected_rerun/ G19 的确定性对照
│   │                     canon/          ✅ hf_canon.parquet（537,025 行）CANON_REPORT.json
│   ├── canon/         hf_canon.parquet
│   ├── ladder/        ✅ 100k_model_ids.csv（100,000 行 / 6.9 MB，sha256 ccf288ae…）
│   │                     LADDER_REPORT.json
│   ├── feats/100k/    x_m.npy  size_bucket_id.npy  family_id.npy  family_vocab.csv
│   └── graphs/        hgraph_ml_v2.pt(CORE)  hgraph_ml_v2_sub.pt(R0)  hgraph_100k.pt
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
├── scale1m/           ✅ __init__.py  hf_crawl.py  query_plan.py  taxonomy.py
│                      ✅ annotate_candidates.py  select_balanced_halo.py
│                      ✅ gates.py  report_distribution.py  verify_raw.py
│                      hf_canonicalize.py  build_ladder.py
│                      embed_lake.py  build_graph_rung.py  verify_rung_graph.py
│                      train_rung.py  export_rung.py  eval_rung.py  write_manifest.py
│   └── tests/         ✅ test_hf_crawl.py / test_taxonomy.py / test_selection.py（**77 passed**）
├── configs/           ✅ halo_quotas.json（全部配额与上限的唯一出处，`halo-quota-1.2`）
│                      watgpu.example.env（仅占位符）
├── scripts/watgpu/    bootstrap_env.sh  smoke_test.sbatch  train_rung.sh
│                      evaluate_rung.sh  build_index.sh  crawl.sh
│                      collect_run_metadata.sh  download_results.ps1
└── docs/1M/           T0.md  T0_runs/  ✅ T2.md  ✅ T2_runs/
    └── S4/            R2_EXECUTION.md
```

> `verify_raw.py` 是计划原表里没有的一个模块，实际是必需的：它同时是 T2 的出闸门和 T3 的入口守卫，
> 沿用 `scale/pull_corpus.py` + `scale/verify_corpus.py` 的成对体例（pull 冻结 / verify 守卫）。

---

## §15 命令速查（按顺序）

```bash
# ── 本地：T0 改造与验收 ─────────────────────────────────
.\.venv\Scripts\python.exe -m pytest ModelLakeFishing/stage2TrainGraphSAGE/tests -k "fanout or sparse_M or chunked"
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.export_ours `
  --graph .../hgraph_ml_v2_sub.pt --seed 0 --epochs 25 --tag R0_anchor_post
#   ⇒ gold@10 必须 = 0.4159 ± 噪声

# ── watGPU：T1 探测 ──────────────────────────────────
sinfo -o "%P %N %G %m %c %l"; quota -s
srun --gres=gpu:1 --time=00:05:00 --pty bash -c 'curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models?limit=1'
git pull --ff-only
git rev-parse HEAD; git status --short

# T2–T8 的 python 命令写入对应 sbatch wrapper，并由 srun 在计算节点执行。
# 登录节点只负责同步、轻量检查、提交与监控。

# ── T2 v2 ✅ 已完成（本地，2026-08-03）─────────────────
# HF_TOKEN 可选（匿名限流 500req/300s，limit=1000 每页）
C=$MLF_DATA_DIR/data1m/candidates_v2
python -m scale1m.hf_crawl --plan full                              # 359,388
python -m scale1m.hf_crawl --plan deep --out $C                     # 492,391
python -m scale1m.hf_crawl --backfill-deficits $C/selected/DEFICITS.json --scale 0.3 --out $C   # 537,025
python -m scale1m.annotate_candidates --candidates $C
python -m scale1m.select_balanced_halo --annotated $C/annotated.parquet --target 69817 \
       --exclude-core stage1BuildTransferGraph/hgraph_ml_v2.pt      # 🔴 D-35
python -m scale1m.select_balanced_halo --annotated $C/annotated.parquet --target 69817 \
       --exclude-core stage1BuildTransferGraph/hgraph_ml_v2.pt --out $C/selected_rerun --quiet   # G19
python -m scale1m.verify_raw --selected $C/selected/selected_halo.parquet \
       --annotated $C/annotated.parquet --rerun-check $C/selected_rerun/selected_halo.parquet    # G1-G20
python -m scale1m.report_distribution --v1 .../annotated_v1.parquet --pool $C/annotated.parquet \
       --selected $C/selected/selected_halo.parquet --out docs/1M/T2_runs --tag v2
python -m pytest scale1m/tests -q                                   # 77 passed

# v1 快照守卫（raw/ 仍只读、仍可验）
python -m scale1m.verify_raw --core stage1BuildTransferGraph/hgraph_ml_v2.pt --sample 20 --forecast 100000

# ── T3 梯子 ✅ 已完成（本地 CPU，2026-08-03）──────────
python -m scale1m.hf_canonicalize --candidates $C --core .../hgraph_ml_v2.pt   # ~3 min
python -m scale1m.build_ladder --rung 100k --n 100000 --core .../hgraph_ml_v2.pt \
       --halo $C/selected/selected_halo.parquet --canon $C/canon/hf_canon.parquet \
       --out .../ladder                                                        # ~20 s
python -m pytest scale1m/tests -q                                              # 89 passed

# ── T4 特征 ─────────────────────────────────────────
sbatch --gres=gpu:1 --mem=32G --time=1:00:00 --wrap \
  "python -m scale1m.embed_lake --rung 100k --out \$WORK/model_lake/data1m/feats"

# ── T5 建图 + 验图 ───────────────────────────────────
python -m scale1m.build_graph_rung --rung 100k --core ... --ladder ... --feats ... --out .../hgraph_100k.pt
python -m scale1m.verify_rung_graph --rung 100k          # 🔴 全绿才往下

# ── T6 训练（三步，不许跳）────────────────────────────
RUNG=12k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # 锚点
RUNG=30k  SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # D-9 解除
RUNG=100k SEED=0 EPOCHS=2  sbatch scripts/watgpu/train_rung.sh   # 冒烟
RUNG=100k SEED=0 EPOCHS=25 sbatch scripts/watgpu/train_rung.sh   # 全量

# ── T7 导出 + 索引 ──────────────────────────────────
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

## §16 本档新增决策登记（续 `plan.md` D-17）

| ID | 决策 | 定案 | 理由 |
|---|---|---|---|
| **D-18** | HALO 用哪个描述符函数 | **CORE 的 `modellens_build_graph.model_descriptor(mid, family, size_b)`**，HF 富字段丢弃 | 信息量必须对齐，否则两群平凡可分、gold@10 假性不降。报告须明写「主动放弃 HF 更丰富元数据是受控实验的要求」 |
| **D-19** | CORE 特征是否重算 | **不重算，原样复制** | MiniLM 跨设备末位差异会让 CORE 不再是 P3/P5 的 CORE，五档梯子断链 |
| **D-20** | 爬取是否一次到 1M | **不**。按 downloads 降序、可续、只追加；R2 只取前 15 万 | 产物天然是前缀嵌套，R3/R4 续爬即可，不重爬。**T2 补充（F-T2-9）**：游标是建在可变字段 `downloads` 上的 keyset cursor ⇒ 跨周续爬只是**近似**嵌套；按 id 去重强制、每档 `snapshot_date_utc` 分别披露 |
| **D-21** | `plan.md §4` 的第 5/9/10 项 | **100K 档不做**，留 R3/R4 | 数据集数不变、fp32 装得下、图 ~250MB。不提前优化 |
| **D-22** | R2 的 seed 数 | **3 seeds**（0,1,2） | 100K 单次仅 ~30 min，没有省的理由；R3/R4 才考虑降到 1 |
| **D-23** | watGPU 路径与存储 | **集中配置；首次登录实测后填写** | 集群存储、quota 与授权目录不能由本地路径或通用集群惯例推断 |
| **D-24** | 计算执行位置 | **登录节点仅做轻量控制；冒烟用短时 allocation；正式运行用 `sbatch`** | 资源密集阶段必须由 Slurm 调度并留下资源与作业记录 |
| **D-25** | preemption/requeue 身份 | **沿用 logical `RUN_ID`，恢复完整训练状态；新实验分配新 `RUN_ID`** | Slurm job ID 可能变化，实验身份与科学记录需要保持连续 |
| **D-26** | HALO 的 `size_b` 是否用名字正则兜底 | 🔴 **不用。只取 `safetensors.total`** | F-T2-3 实测：CORE 47.91% 带 size 子句，HALO 仅 safetensors = 38.17%（差 −9.7 pp），加正则 = 78.60%（差 +30.7 pp）。D-18 的「信息量对齐优先于信息量最大化」在这里落成数字。**主动放弃 62,813 条可从名字恢复的参数量，报告须明写** |
| **D-27** | `dropped` 层判据恒为假怎么办 | **保留判据不动，报告里写明它恒为 0** | F-T2-7：HF 自动挂 `region:*`/`license:*` 标签 ⇒ `tags` 覆盖率 100% ⇒ 合取式永假。`layer` 只进审计不进训练（D-10），恒空无下游影响；改判据 = 引入一条未验证的新过滤规则，风险大于收益 |
| **D-28** | 爬到的 1,199 个 CORE 模型的 `base_model` 用不用 | 🔴 **不用，不给 CORE 子图加任何边** | F-T2-4：用了就是改动 CORE，R0/R1/R2 之间「N 是唯一变量」当场破功。这是明确的不作为，不是遗漏 |
| **D-29** | T2 是否必须在 watGPU 上跑 | **否，本地跑并上传** | F-T2-12：产物 24.8 MB / 46.8 秒，传输秒级。D-12 的外网分叉保留为 R4 的优化项，不再阻塞 T1/T2 |
| 🔴 **D-30** | HALO 总体怎么定义 | **不是「downloads 榜前 N」。爬取与选样彻底分离**：多源发现 → 标注 → 家族/近重复归并 → 配额分配 → 确定性选样 → 缺口回补 → 冻结。全局 downloads 排名只保留为**候选发现的一路信号**与**分层内的流行度信号** | v1 实测：一个发布者 24.21%、量化搬运 61.98%、text-generation 50.77%、任务熵 0.601。这是采样机制的产物，不是样本量问题。v2 后同口径为 0.90% / 15.00% / 18.00% / 0.917 |
| 🔴 **D-31** | 百分比上限的分母 | **实际选出的总体，不是 requested target**。上限按工作规模 N 参数化，**二分搜索满足 `achieved(N) ≥ N` 的最大 N**；出闸门再按真实计数独立复核 | 用 target 当分母 ⇒ ①把 target 写大就能买到更宽的上限 ②跑不满时按实际产出算已越界。初版还配了向下自适应循环，会把可行问题误判为不可行（69,817→…→60,202） |
| 🔴 **D-32** | 「英语侧 ≥75%」的分母 | **可归属语种的总体**（除 `language-neutral` 外的全部）。`language-neutral` 另有自己的全体份额上限 | 把 ViT 计入英语份额不测量任何东西，且与任务均衡直接冲突（语种无关的 5 个 supertask 在任务配额下合计就要 >25%）。实测：按全体作分母时卡在 ~60,200；改分母后 69,817 一次探测即可行 |
| 🟠 **D-33** | 官方量化与第三方搬运 | **分成两类 source type**（判据：量化仓库作者是否 == base 作者）。15% 硬顶落在**第三方**类上，两者合计另设 20% 硬顶 | `Qwen/Qwen3-8B-GGUF` 与 `somebody/Qwen3-8B-i1-GGUF` 不是同一类总体；合并计数要么丢掉正当的第一方发布，要么让搬运方搭便车。合计上限防止「拆分」被用来抬高总量 |
| 🟠 **D-34** | 任务份额是目标还是硬顶 | **两者都要**：15% 是配额分配的目标，18% 是任何阶段不可越过的硬顶，G8 按实际总体检查 | 只有目标 ⇒ 放松阶段无界；只有硬顶 ⇒ 配额分配失去梯度 |
| 🔴 **D-35** | 选样前是否剔除 CORE | **必须剔除**（`--exclude-core`） | F-T2-23：候选池按任务/语种发现自然会捞到 CORE 成员（池中 7,372 条）。不剔除时 69,817 条里有 1,892 条已在 CORE ⇒ `CORE ∪ HALO` 只有 98,108 个 distinct，而 R2 定义是 100,000。**G1–G20 一条都不会报警**（G1 只查 HALO 内部重复）。剔除后实测 `∩=0`、`∪=100,000` |
| 🔴 **D-36** | HALO 的 `family` 怎么推 | **四级解析，目标是 CORE 的 `config.model_type` 命名空间**：① model_type 且 CORE 用过 ② 小写化规则输出且 CORE 用过 ③ model_type ④ 小写化规则输出。每行记 `family_source` | F-T3-1：计划原文的 `_infer_one_family` 只得 **0.64%** 的 CORE-used 命中（返回 `LLaMA`，CORE 用的是 `llama`，vocab 里是两行）⇒ 同架构两行嵌入、零共享、不报错。四级解析升到 **48.47%**。附带纠正：该函数**没有** tags 兜底 |
| 🔴 **D-37** | `lineage_base` 取哪个字段 | **`baseModels` 优先，`cardData.base_model` 兜底**，并保留 `relation` | F-T3-4：537K 候选上 `baseModels` 命中 234,445、`cardData.base_model` 只有 12,713 ⇒ 只用后者会丢 **94.9%**。`relation`（quantized/adapter/finetune/merge）正是 `r_mm'` 离散权重需要的量，T5 不必从名字猜 |
| 🟠 **D-38** | 梯子的 CORE 侧填不填四个量 | **留空，`layer="core"`** | 铁律 1：CORE 这些量从冻结图原样复制、绝不重算。梯子在 CORE 侧只钉 id 与行序，T4 直接读 `hgraph_ml_v2.pt` |
| 🟠 **D-42** | §9.1 列的 `--amp` 做不做 | **R2 不做，推到 R3/R4 显存真的不够时再加，且加的时候单独跑一次锚点对拍** | T6 第 1 步的 12K 锚点要证明「换了环境、开了 T0 四个开关，模型还是原来那个」；同一批改动里既换环境又换数值精度，锚点一偏就无法归因。实测佐证：12K 峰值 0.583 GB、100K 峰值 1.737 GB，而卡是 139.8 GB 的 H200 NVL ⇒ **AMP 现在解决的不是一个我们有的问题**。见 [`T6GPU.md`](T6GPU.md) §8 问题 1 |
| 🔴 **D-43** | `--seed` 的语义与锚点噪声带 | **`--seed` 是 `split_seed`，`init_seed` 恒为 0**（沿用 `export_ours.py`，不改）。因此多 seed 给的是**划分噪声**区间，**不是初始化噪声**；报数字时必须写明是哪一种 | T6 实测：三个 100K seed 的 held-out 数据集数是 734 / 775 / 936，`tau_macro` 0.180 / 0.077 / 0.217（散布 2.8×）—— 三个 run 的测试集根本不是同一个。**连带发现**：G-C2 引用的「`0.4159 ± 3-seed 噪声`」里那条噪声带**从未被测过**——`P3_EXECUTION.md` 的 0.4159 是 `--seed 0` 的**单次**结果（517 个 held-out 数据集），P4/P5 复用同一个数。判据必须先补测再使用，见 [`T6more.md`](T6more.md) P1-3 |

---

## §17 与 `plan.md` 的对应关系

| `plan.md` 阶段 | 本 runbook | 100K 档的裁剪 |
|---|---|---|
| S0 环境 | T1 | access、存储、同步、环境、smoke、tiny `sbatch` 全部过 gate；R2 初值 `--mem 96G` / `--time 04:00:00` 由实测修正。**不再是 T2 的前置（D-29）** |
| S1 数据 | T2–T5 | 图不拆件（D-21）。**T3 ✅ 已完成**（本地 CPU，梯子 100,000 行、五门全过）。**T2 ✅ 已完成 v2**（本地，2026-08-03）：532 个查询 → 537,025 候选 → **配额选出 69,817**，18/20 硬门过，与 CORE 零交集。v1 的 15 万条降级为审计件（D-30） |
| S2 代码 | T0 | 10 项做 7 项（D-21） |
| S3 作业化 | T1 + T6 | 训练与建索引**不拆作业**（100K 的索引 CPU 开销还小）；batch metadata、checkpoint/resume、监控与回收形成闭环 |
| S4 执行 | T6 | R0 → R1 → R2 三档 |
| S5 评测 | T8 | A/B/C/D 四轴全做；曲线拟合只有 3 点，**不下结论，等 R3/R4** |
