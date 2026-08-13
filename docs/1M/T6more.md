# T6 收尾：进 T7 之前要闭合的事

上级 runbook：[`100kplan.md`](100kplan.md) §9 / §12。上游执行记录：[`T6GPU.md`](T6GPU.md) §12（工作流 `T6_20260811T174245Z`）。

**一句话：训练本身合格，六个 run 的机制门都是真的过了；卡住 T7 的不是训练，是三条证据链。**

> **执行状态（2026-08-13，远端已执行）**：`§3.1` 与 `§3.2` **两条阻塞已全部闭合**，
> 本地 9 项早先已完成（§2）。`§3.3` 的 vocab 已在远端产出并逐位对上，
> **12K 的两个 seed 已提交、在排队**（`Reason=Priority`，watgpu808 两张 GPU 满，预约起跑 `19:28:55`）。
> 唯一新增的红旗是 **§1.8：远端正式图的血缘 `relation_id` 全是 `unknown`**——需要你定一次。
> 所有已确定的数字与指纹在 **§1 指纹台账**，那一节是本篇给后续阶段的主要交付。

---

## 0. 执行状态总表

| # | 项 | 状态 | 在哪看 |
|---|---|---|---|
| P1-1 | 100K 图的 T5 门禁留痕 | ✅ **已闭合**：sha 对上，33/33 门全过 | §3.1 |
| P1-2 ① | 远端 as-run 代码与本地对账 | ✅ **已闭合**：3 处异常全部归因，均无害 | §3.2 |
| P1-2 ② | 提交 T6 源码树 | ✅ 完成 `3992d67` | §2.1 |
| P1-2 ③ | 修 `_git()` 的 cwd + 回归测试 | ✅ 完成，7 个新测试 | §2.2 |
| P1-3a | seed 语义定案 | ✅ **已查清并登记 D-43** | §2.3 |
| P1-3b | 12K 补 seed 1/2 | 🟡 **已提交，排队中**（1510204 / 1510205） | §3.3 |
| P2-1 | 12k/30k 的 `family_vocab.csv` | ✅ 完成，**远端产物与本地逐位相同** | §2.4 / §3.3 |
| P2-2 | override 写进 gate report / SUMMARY | ✅ 完成，端到端验过 | §2.5 |
| P2-3 | run 目录复用登记 | ✅ 完成 | §2.6 |
| P2-4 | `checkpoints` 字段语义 | ✅ 完成 | §2.6 |
| P2-5 | `--amp` 决策登记 | ✅ 完成 **D-42** | §2.7 |
| ＋ | **新发现 1**：CRLF 污染 vocab 指纹 | ✅ 已修 + 已加 `.gitattributes` | §2.4 |
| ＋ | 🔴 **新发现 2**：远端图的血缘 `relation_id` 全 `unknown` | ⏸ **待你裁定** | §1.8 |

全套测试：**206 passed**（`scale1m/tests` 140 + `stage2TrainGraphSAGE/tests` 66），改动后重跑通过。

---

## 1. 指纹台账

**这一节是本篇最重要的交付。** 下面每一个数都是实测复算出来的，不是抄的；
带 ✅ 的是我在本地独立重算并比对通过的。后续任何一步发现对不上，先回来查这张表。

### 1.1 三档图

| 档 | `graph_sha256`（T6 六个 run 绑定的值） | 本地对应文件 | 核验 |
|---|---|---|---|
| 12k | `00ac2434cd8ecaa2f633bc377b25d1d5ae0a7d50cbcab88774355efc47a11d34` | `stage1BuildTransferGraph/hgraph_ml_v2_sub.pt` | ✅ **逐字节相同** |
| 30k | `e6ae2dfeb59779f4cb0242a08bf90d6628699e90a9a648048b799d9cc7d71cda` | `stage1BuildTransferGraph/hgraph_ml_v2.pt` | ✅ **逐字节相同** |
| 100k（8-11 训练用，**已归档**） | `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0` | 远端 `hgraph_100k_norel_7fbc3c47.pt` | ✅ 远端实测相同；血缘无关系类型（§1.8） |
| 100k（**现行**，带关系类型） | `16f3521482a8efd69d491b66f84624d9e0086f3cb42288d69a9c7f022492060f` | 远端 `hgraph_100k.pt` | ✅ 2026-08-13 重建，33/33 门过（§3.4） |

> 12k/30k 这两行是**铁律 1 在这两档上的直接证据**：训练用的图与本地冻结的 CORE 图逐字节相同。
> 100k 这一行现在也闭合了：远端实测 `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0`，
> 与六个 MANIFEST 绑定的值逐位一致，且这张图的 T5 门禁 33 条全过（§3.1）。
> 远端 12k/30k 是指向 CORE 图的**软链**（`ls -la` 已确认），所以三档的图身份全部可追。

### 1.2 100K 图：本地排练件 vs 远端正式件（**已实测对账**）

| 量 | 本地排练件（2026-08-10T17:51:36Z） | 远端正式件（T6 实际训练用，2026-08-10T21:44:40Z） | 判定 |
|---|---|---|---|
| `graph_sha256` | `334153aac46d705ccab79012bdbd30402b863ac9aedee2752ca69f67d32ea004` | `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0` | 预期不同 ✅ |
| `x_m_sha256` | `6e8b69245a462cd5299a10c49565784c749ab5d23d6314934557948901ba7fd3`（对照） | `d7573a020779e69c082cf686933204d5fe3261960fe4e1525f5a264596721da5`（正式） | 预期不同 ✅ |
| `family_vocab_sha256` | `0fbf5c11d3cae2ea8254ee8ac218babe158c0c932a43c4ce002d7a64dc4aa157` 🔴 CRLF | `f40af358d64d96a2a7f7f3e2ff4e323abf1b28c8372f09ddc2e257cf36d9f31d` LF | 预期不同（换行符）✅ |
| `core_graph_sha256` | `e6ae2dfeb59779f4cb0242a08bf90d6628699e90a9a648048b799d9cc7d71cda` | **相同** | ✅ |
| `ladder_sha256` | `ccf288aee2a3caee10b60ab8a7ee3031f44d5766e33074e4090eb35cb128b0ab` | **相同** | ✅ |
| 边数 / 血缘拓扑 | 见 §3.1 | **逐项相同** | ✅ |
| `lineage.relation_counts` | `finetune 10226 / quantized 4360 / adapter 1608 / unknown 115 / merge 76` | 🔴 **`unknown 16385`（全部）** | **第四处不同 → §1.8** |

> ✅ **本篇写在执行之前的那条预判被实测证实了**：T5.md §8 说「**只有** `graph_sha256` 和 `x_m_sha256` 会不同」，
> 而 `family_vocab_sha256` 也不同，原因与科学无关——纯粹是换行符（§2.4）。**这一条已从"预测"升为"实测"。**
>
> 🔴 **但实测同时抓出了第四处不同，而本篇定的规矩正是「第四处不同才是问题」。** 见 §1.8。

### 1.8 🔴 待裁定：远端正式图的血缘 `relation_id` 全部是 `unknown`

**事实。** 两份 100K 图的血缘**拓扑完全一致**——`total 16427 = core_core 42 + core_halo 606 + halo_core 0 + halo_halo 15779`，
`n_models_with_lineage 20135`、`n_components 3709`、`largest_component 159`、
`halo_declared_base 27672`、`halo_base_unresolved 10887`、`halo_self_loop_dropped 400`，**逐项相同**。
唯一的差别是每条边的 `relation_id`：

| | 本地排练件 | 远端正式件（T6 训练用） |
|---|---|---|
| `relation_counts` | finetune 10226 / quantized 4360 / adapter 1608 / unknown 115 / merge 76 | **unknown 16385**（+ CORE 的 42） |

**成因已查实**：远端根本没有 `hf_canon.parquet`——`$DATA_ROOT/data1m/candidates_v2/canon/` **目录不存在**。
[`T5.md`](T5.md) §8 明确预告过这件事（「不传也能跑，代价是所有血缘边的 `relation_id` 落到 `unknown`，以后要用只能重建图」）
并给了 `scp` 命令，那条命令没有被执行。

**对已有结果的影响：无。** `relation_weights_applied` 在**两份图上都是 `false``（D-41：加权是显式开关，默认关），
`edge_attr` 两边都是 CORE 的 1.0，边的**存在与拓扑**完全相同 ⇒ **T6 六个 run 的数字不受影响，不需要重跑**。

**丢的是什么**：`r_mm'` 的离散有序权重（quantized > adapter > finetune > merge）在这张图上**没有输入**。
那是 CLAUDE.md 点名的 novel contribution，也是 D-37 花力气从 `baseModels` 抢回 94.9% 血缘声明的目的。

**✅ 裁定：走 B（用户 2026-08-13 定），已执行。** 见 §3.4。

| | 做法 | 结果 |
|---|---|---|
| A（保守） | 不重建，R2 如实披露「血缘无关系类型」 | 未采用 |
| **B（采用）** | 传 parquet → 重建 100K 图 → 重跑 100K 档的 run | ✅ 图已重建并验过；4 个 run 已提交 |

> **只重跑 100K 档，不动 12k/30k。** 那两档绑的是 CORE 图（`00ac2434…` / `e6ae2dfe…`），
> 本次重建一个字节都没碰它们，所以它们的 checkpoint 与 binding 依然成立。

### 1.3 family_vocab（checkpoint 的身份凭据）

| 用于 | 行数 | sha256 | 来源 |
|---|---:|---|---|
| **12k / 30k**（CORE，341 家族） | 341 | `d4e6b6823edcc5b76f797479ceaf27aec0ead552cac18bcc5f60012b0935ec54` | ✅ 本篇新生成；两张 CORE 图各自导出**同一份文件** |
| **100k**（T4，1896 家族） | 1896 | `f40af358d64d96a2a7f7f3e2ff4e323abf1b28c8372f09ddc2e257cf36d9f31d` | ✅ = T6 三个 100K seed 绑定的值；= 远端 `ckpt/family_vocab.csv` 的 24,281 字节 |
| ~~100k 的 CRLF 副本~~ | 1896 | ~~`0fbf5c11d3cae2ea8254ee8ac218babe158c0c932a43c4ce002d7a64dc4aa157`~~ | ❌ 内容相同、仅换行符不同；**已归一化为 LF** |

**一条独立交叉验证**：用新写的 [`scale1m/dump_family_vocab.py`](../../scale1m/dump_family_vocab.py) 从**本地排练件** `hgraph_100k.pt`
导出 vocab，得到的 sha 是 `f40af358d64d96a2…`——**与远端正式件绑定的值逐位相同**。
这同时证明两件事：① 导出器的格式与 T4 的产物字节一致；② 远端正式图与本地排练图的 **family 命名空间是同一个**
（两份图只在特征浮点值上不同，vocab 完全一致）。在 100K 图本体还没核验之前，这是我们能拿到的最强旁证。

### 1.4 交付包完整性 ✅ 本地全部复算通过

```text
delivery  ec79b68167966b5a669f8535b6a9d7c0500673043614605993d7ff17d8ed3856  T6_20260811T174245Z_delivery.tgz
artifact  c63712f2bdfe689d8fdcaef92bafd610f8538db1ca70e25b39b6702b0c2ce213  t6_R2_12k_s0_e25.tgz
artifact  85009833e87f21a12a0060366014e4a0e84d51256381ad031508a8caa2b4b6c3  t6_R2_30k_s0_e25.tgz
artifact  baec1186622bb34990fc2d725e6898bb2210fa967256f29c37b857f6479647ae  t6_R2_100k_s0_e2.tgz
artifact  03897aaafcb228ad77329ea9818564645d0542db897e3cc321bf4be9aabf670d  t6_R2_100k_s0_e25.tgz
artifact  68da6e136261676e8f7755acace5ba43cd9a6709063cb508816d31c6a733deaa  t6_R2_100k_s1_e25.tgz
artifact  7a4c57412badd1d922b8aa5e15de8beab1284e423b1980e71cc2a8dfc3d49d46  t6_R2_100k_s2_e25.tgz
```

### 1.5 六个 run 的关键数（`MANIFEST.json`）

| run | N | expect_n | ep | 墙钟 | 峰值显存 | loss first → last | 机制门 | vocab 绑定 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `R2_12k_s0_e25` | 12,000 | *(null)* | 25 | 93.0 s | 0.583 GB | 19.0257 → 13.8466 | PASS | 🔴 null |
| `R2_30k_s0_e25` | 30,183 | 30,183 | 25 | 105.5 s | 0.866 GB | 18.9898 → 15.0575 | PASS | 🔴 null |
| `R2_100k_s0_e2` | 100,000 | 100,000 | 2 | 14.2 s | 1.719 GB | 19.8590 → 18.2148 | PASS | ✅ f40af358 |
| `R2_100k_s0_e25` | 100,000 | 100,000 | 25 | 108.5 s | 1.737 GB | 19.6944 → 15.4304 | PASS | ✅ f40af358 |
| `R2_100k_s1_e25` | 100,000 | 100,000 | 25 | 116.6 s | 1.713 GB | 19.3075 → 14.8090 | PASS | ✅ f40af358 |
| `R2_100k_s2_e25` | 100,000 | 100,000 | 25 | 106.7 s | 1.723 GB | 19.8016 → 14.8894 | PASS | ✅ f40af358 |

六个 run 共同的：`init_seed = 0`、`resume_mode = fresh`、`start_epoch = 0`、`resumed_from = null`、
`git_head = ""` 🔴、`hostname = watgpu808`、`NVIDIA H200 NVL 139.8 GB`、`torch 2.12.0+cu130`、`python 3.11.9`、
开关 `fanout / sparse_M / contrast_n_neg=256 / chunked_infer=50000`、
配置 `1 layer / top_frac 0.10 / lr 0.01 / λ_rank=λ_contrast=λ_global=1.0 / global_n_neg 256 / batch 1024 / lake α=0.75`。

**✅ 门禁独立复算**：把 [`validate_t6_run.py`](../../scale1m/validate_t6_run.py) 在本地对着这六份 MANIFEST 重跑了一遍，
**6/6 全过**（15 项 checks 各自 true），并复现了 `100k_smoke` 的 25-epoch 外推 `projected_25_epoch_s = 177.5`（远小于 4 h 上限）。

### 1.6 本地代码与提交

| 项 | 值 |
|---|---|
| 分支 | `r2-100k`（远端 `origin/r2-100k`） |
| T6 之前的 HEAD | `b512213` "T4: stage Python environment on node-local filesystem" |
| **本篇提交 1** | **`3992d67`** T0+T6: pin the source tree behind run T6_20260811T174245Z |
| **本篇提交 2** | **`f013027`** Pin LF for the files where line endings change meaning |
| 本地代码指纹清单 | [`T6more_runs/local_code.sha256`](T6more_runs/local_code.sha256)（**115 个文件** + 3 行注释头 = 118 行）<br>自身 sha `c482bdaa1908fcbd81e1241937d7f701163f8fffd7d19f055152537d1d19eace` |
| 测试 | 206 passed（改动后重跑） |

### 1.7 R0 锚点 `0.4159` 的出处（P1-3a 的调查结果）

| 项 | 实测 |
|---|---|
| 出处 | [`docs/scale/P3/P3_EXECUTION.md`](../scale/P3/P3_EXECUTION.md) §2 |
| 怎么产生的 | `scale/export_ours.py --seed 0 --epochs 25`，**单次运行**，L1L3b 未改动配置 |
| 语义 | `--seed` → `split_seed=0`；`init_seed = INIT_SEED = 0`（常量） |
| 评测口径 | 12,000 候选、**517 个 root-aware held-out 测试数据集** / 412 roots，median gold rank 21 |
| 配套数字 | `gold@1 0.0948` / `gold@10 0.4159` / `root-macro 0.3920` / `top3@10 0.5319`；`five_metric_eval == global_metrics`（0.41586 == 0.41586） |
| 🔴 **噪声带** | **不存在。** P3 只跑了一个 seed，P4/P5 复用同一个数（P4 §「我方 0.4159 与 P3 held-out 报告逐位一致」） |

**⇒ G-C2 判据「`0.4159 ± 3-seed 噪声`」引用了一个从未被测量过的量。**
补测的办法在 §3.3；语义已登记为 **D-43**。

**一条对上了的旁证**：T6 的 `R2_12k_s0_e25` 的 `train_row.head.n_datasets = 517`——
与 P3 的 517 完全一致 ⇒ **T6 的 12K 跑的是与 P3 同一个划分**，锚点具备可比性（前提是噪声带补上）。

---

## 2. 已闭合的（本地）

### 2.1 P1-2 ②　代码树已钉住　✅ `3992d67`

**问题原样记录：** 六个 run 的 `git_head` / `git_status` 全是空字符串，而工作树当时是脏的——
`scale1m/` 下 T6 的全部代码是 untracked，`stage2TrainGraphSAGE/` 的 T0 七项改造是 modified，
HEAD 还停在 T4 的 `b512213`。**产出那六个数字的代码一行都没进版本库。**

提交前做了安全检查：69 个文件、docs 617 KB，**无 `.pt`/`.parquet`/`.gz`/`.npy` 等数据件、无 token / 私钥 / 明文口令**；
`configs/` 下只有 `halo_quotas.json`（T2 的配额配置，无凭据）。

> 🔴 **诚实标注：这个提交是「as-run 树 **+** 本篇的修复」，不是纯粹的 as-run 树。**
> 本地已经没有 as-run 的字节了（那些文件当时不在版本控制里，我在同一份文件上就地做了修复）。
> 唯一还留着 as-run 副本的地方是**远端的 `$PROJECT_ROOT`**。
> 所以 §3.2 的对账不是走形式：它跑完之后，差异**应当恰好**是 `3992d67` 提交信息里列的那 9 个文件，
> **多出任何一个差异都是未解释的，必须查**。

### 2.2 P1-2 ③　`_git()` 的根目录已修　✅

根因是一行路径：[`train_rung.py`](../../scale1m/train_rung.py) 的 `_REPO_ROOT` 算成了 `scale1m/` 的祖父目录
（`codes/`，那里有个 git 不认的空 `.git` 空壳），而仓库根其实是 `codes/ModelLakeFishing/`。

改动三处，都在 [`train_rung.py`](../../scale1m/train_rung.py)：

1. 新增 `_GIT_ROOT = os.path.dirname(_HERE)`，**`_REPO_ROOT` 保持不动**——它担着 `sys.path`，
   `import ModelLakeFishing.*` 靠它，动了会 import 失败。
2. `_git()` 改为**失败返回 `None`**，成功才返回 stdout。理由写在代码注释里：
   空串是 `status --short` 干净时的**合法结果**，把失败也折叠成空串，等于让「仓库坏了」长得和「树是干净的」一模一样——
   这正是这个 bug 藏了整整一轮的原因。
3. git 不可用时写入 `git_head = git_status = "not-a-git-repo"`，**大声说不知道**，而不是装作干净。

新增 [`scale1m/tests/test_run_metadata.py`](../../scale1m/tests/test_run_metadata.py)，**7 个测试**，其中三个直接对着这个 bug：

- `test_git_root_is_the_repository_not_its_parent` —— 钉住 `_GIT_ROOT != _REPO_ROOT` 的关系
- `test_git_head_is_recorded_when_running_inside_the_checkout` —— 在 checkout 内必须拿到 40 位 hex
- `test_git_failure_is_distinguishable_from_a_clean_result` —— **失败不许长得像干净**

### 2.3 P1-3a　seed 语义定案　✅ 登记 **D-43**

查清了三件事，全部写进 `100kplan.md` §16 的 D-43：

1. **`--seed` 是 `split_seed`，`init_seed` 恒为 0。** 这是忠实沿用 [`export_ours.py`](../../scale/export_ours.py)，**不是 bug，不改**。
2. **所以多 seed 给的是「划分噪声」，不是「初始化噪声」，三个 run 的测试集根本不是同一个**：

   | run | held-out 数据集数 | `tau_macro` | `mean_cos` |
   |---|---:|---:|---:|
   | `R2_100k_s0_e25` | 734 | 0.180 | 0.729 |
   | `R2_100k_s1_e25` | 775 | 0.077 | 0.712 |
   | `R2_100k_s2_e25` | 936 | 0.217 | 0.767 |

   `tau` 散布 2.8 倍。这不是坏事（划分噪声比初始化噪声更能说明问题），**但报出去时必须写明是哪一种**。
3. **`0.4159` 从来没有噪声带**（详见 §1.7）。判据必须先补测再使用。

### 2.4 P2-1　family_vocab 已补齐，并抓到一条 CRLF 陷阱　✅

**做了什么。** 新增 [`scale1m/dump_family_vocab.py`](../../scale1m/dump_family_vocab.py)：
从任意一档图的 `xm0_meta["family_vocab"]` 导出 `family,family_id` 的 CSV，按 id 排序，
断言 id 是连续的 `0..N-1`（否则嵌入表第 k 行与文件第 k 行会指不同的家族），
**显式写 LF**，并同时打印图的 sha 与文件的 sha。

跑出来的结果在 §1.3：两张 CORE 图导出的是**同一份 341 行文件**（`d4e6b682…`），
已放到 `data1m/feats/{12k,30k}/family_vocab.csv`——就是 [`train_rung.sbatch`](../../scripts/watgpu/train_rung.sbatch)
本来就会去找的位置，所以**重跑这两档时会自动绑上，不需要改任何命令**。

**顺手把静默降级堵死。** 那个 sbatch 原本是 `[ -f "${VOCAB}" ] && VOCAB_ARG=(...)`——
文件不在就**悄悄不绑**，这正是 12k/30k 蒙混过关的机制。现在改成**找不到就 `exit 2`**，
并在错误信息里直接给出 dump 命令。CLAUDE.md 把「vocab 与 checkpoint 脱钩」列为三大风险之二，
这种地方不该容忍静默降级。

> 🔴 **新发现：CRLF 会在不改一个字符的前提下改掉指纹。**
> 本地 `data1m/feats/100k/family_vocab.csv` 当时是 **CRLF**（26,178 字节，sha `0fbf5c11…`），
> 而 T6 六个 run 绑定的是 **LF** 形式（24,281 字节，sha `f40af358…`）。
> `diff` 掉 `\r` 之后两者**逐行相同**——内容完全一致，只有换行符不同。
> 成因也查清了：仓库 `core.autocrlf=true` 且**没有 `.gitattributes`**，Windows 上一次 checkout 就会把它写成 CRLF。
>
> 两个动作都已做：① 本地那份**已归一化为 LF**，现在是 `f40af358…`，与绑定值一致；
> ② 新增 [`.gitattributes`](../../.gitattributes)（提交 `f013027`），把 `*.csv` / `*.sh` / `*.sbatch` 钉成 `eol=lf`。
> 后两类是顺带的、同样真实的风险：CRLF 的 shebang 在 Linux 上是 `bad interpreter: ^M`，
> 而且是**排队等到了才失败**。

### 2.5 P2-2　override 现在跟着证据走　✅ 端到端验过

`T6_ALLOW_NO_GOLD_GATE` 原先只落在 `jobs.tsv` 和 `workflow.env`，
六份 gate report 和 `SUMMARY.json` 里没有——与 [`T6GPU.md`](T6GPU.md) §11 的描述不符。已补：

- [`validate_t6_run.py`](../../scale1m/validate_t6_run.py) 新增 `--gold-gate-override {0,1}`，写进每份 report
- [`t6_gate_and_advance.sbatch`](../../scripts/watgpu/t6_gate_and_advance.sbatch) 透传该值
- [`t6_summary.sbatch`](../../scripts/watgpu/t6_summary.sbatch) 写 `gold_gate_override` 字段，
  并在 `scientific_status` 末尾追加 `(advanced under T6_ALLOW_NO_GOLD_GATE=1)`

**验过的方式**：用六份真实 MANIFEST 重跑 gate（6/6 passed，override 记为 1），
再把 summary 的 python 段抽出来跑，输出
`passed True | override True | status: … (advanced under T6_ALLOW_NO_GOLD_GATE=1)`，`STATUS=TRAINING_COMPLETE`。

> 已冻结的 `T6_20260811T174245Z/SUMMARY.json` **不追改**。它是冻结件；更正写在这里。

### 2.6 P2-3 / P2-4　run 目录复用与 checkpoint 字段　✅

`make_run_dir()` 现在返回 `(out, prior)`，`prior` 列出复用前就存在的东西
（非空 `MANIFEST.json` / 非空 `stdout/train.log` / `ckpt/*.pt`），MANIFEST 新增两个字段
`run_dir_reused` 与 `run_dir_prior_contents`，日志也会打印一行。

这条针对的是真事：`R2_12k_s0_e25/stdout/train.log` 顶部有一段**上一次尝试**的 traceback
（`ModuleNotFoundError: matplotlib`，`srun: task 0: Exited with exit code 1`），
而本次工作流的 `logs/train-12k-1508297.out` 是干净的 ⇒ 目录被复用、日志是追加写。
本次 `resume_mode=fresh` / `start_epoch=0`，**数字没有被污染**，但这件事当时在 MANIFEST 里没有任何痕迹。

配套 4 个测试：新目录不报警、零字节日志不算复用（作业脚本会先建 `stdout/`）、
留下的日志要被报出来、留下的 `.pt` 要被报出来。

`checkpoints` 字段拆成 **`checkpoints_written`**（写过的）与 **`checkpoints_retained`**（还在盘上的）。
原字段取的是 `state["saved"]`，而 `keep=3` 会让 `_prune` 剪掉 epoch 4/9——
所以那六份 MANIFEST 里列的 5 个文件，实际只剩 3 个 + `last.pt` / `best.pt`。

**另一条要写进 `R2_EXECUTION.md` 的过程教训**：matplotlib 是在 probe 阶段被**自动补装**的
（`logs/probe-1508294.out` 有完整的 pip 输出），说明 T1 的环境规格不全，**G-W2 该拦没拦到**。

### 2.7 P2-5　`--amp` 已登记为 **D-42**　✅

写进 `100kplan.md` §16。定案：R2 不做，推到 R3/R4 显存真的不够时再加，且加的时候单独跑一次锚点对拍。
实测佐证已一并记入：12K 峰值 0.583 GB、100K 峰值 1.737 GB，而卡是 **139.8 GB** 的 H200 NVL ——
**AMP 现在解决的不是一个我们有的问题**。

---

## 3. 远端执行记录（2026-08-13）

SSH 经 Windows ssh-agent 打通（Git Bash 的 ssh 走不到 Windows 的 agent，必须用
`C:\Windows\System32\OpenSSH\ssh.exe`；这一条留给下次）。执行顺序严格按 §3.1 → §3.2 → §3.3，
**`/tmp/remote_code.sha256` 在任何 `git pull` 之前生成**。

### 3.1 P1-1　100K 图的 T5 门禁留痕　✅ **已闭合**

三份 T5 报告**本来就在远端** `$DATA_ROOT/data1m/graphs/`（`GRAPH_REPORT_100k.json` /
`T5_GATES_100k.json` / `lineage_stats.json`，均为 2026-08-10 21:44 建图时产出），
所以**没有重跑 `verify_rung_graph`，更没有碰 `build_graph_rung`**——直接回收比对即可。

| 检查 | 结果 |
|---|---|
| `sha256sum hgraph_100k.pt` | ✅ `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0`（223,352,080 字节） |
| `T5_GATES_100k.json` | ✅ `passed: true`，**33 条 checks 逐条 `ok: true`** |
| G-B1 铁律 1 | ✅ `model.x[:30183] byte-identical to CORE`、`size_bucket_id/family_id/unique_model_id prefix identical`、`dataset.x untouched`、`trained_on edge_index+attr identical to CORE` |
| G-B2 契约 | ✅ `check_stage2_contract passed` |
| G-B3 划分 | ✅ `test-edge set identical to CORE's`，**65,968 边**，sha `66c81823cb731063614529c73ab142471989858d2a20fadbd537087b8177dcd3`（`rung_sha256 == core_sha256`） |
| 边数 | ✅ `trained_on 312986 / rev 312986 / similar_to 192060 / is_base_of 16427 / rev 16427` |
| 家族表 | ✅ `family_vocab only grew (341 -> 1896)`、`no CORE family row moved`、`ids contiguous 0..1895` |
| 血缘拓扑 | ✅ `16427 = 42 + 606 + 0 + 15779`，与本地排练件逐项相同 |

**三处预期不同全部对上**（§1.2），**但抓到了第四处**——血缘 `relation_id` 全 `unknown`，
这正是本篇立的规矩要拦的东西，已单列为 **§1.8 待裁定**。

远端回收到本地：`D:\research\model_lake\data\runs\T6more_20260813\t5_remote\`

```text
1450a4d5cc40c258a120ff5bbd704af43c268b7ddfd1bd51c077135fe15ea591  GRAPH_REPORT_100k.json
b3bc1a42f680aeca5f8a1e27480b0018c1b65c7d424683d71e625d7014ab8b86  T5_GATES_100k.json
74730705f960f0917e9b7e51368f4f191cafac56d48a03c16b774c6ce75cdfc7  lineage_stats.json
```

### 3.2 P1-2 ①　远端 as-run 代码对账　✅ **已闭合**

`/tmp/remote_code.sha256`：**111 个文件**，自身 sha `e241998d8f9aa8e638205483d6c0452e34bb29900a0656e8355b1b6531850ec5`。
远端 `HEAD = b512213`（与 T6 当时一致），工作树 8 个 modified + 一堆 untracked，与预期相符。

对账结果：**105 个文件逐字节相同**，差异分三类，**全部归因、全部无害**：

| 类 | 文件 | 判定 |
|---|---|---|
| **预期**（本篇改的 5 个） | `train_rung.py`、`validate_t6_run.py`、`train_rung.sbatch`、`t6_gate_and_advance.sbatch`、`t6_summary.sbatch` | ✅ 与 `3992d67` 提交信息一致 |
| **预期**（本篇新增 2 个） | `dump_family_vocab.py`、`tests/test_run_metadata.py` | ✅ 仅本地有 |
| ⚠️ **意外 1** | `scale1m/tests/test_validate_t6_run.py`、`stage2TrainGraphSAGE/tests/test_t0_scale.py` 仅本地有 | 远端从来没同步过这两个**测试**文件；作业不跑 pytest ⇒ **不在运行路径** |
| ⚠️ **意外 2** | `stage2TrainGraphSAGE/w1_dzero.py` 字节不同 | **纯换行符**：本地去掉 `\r` 后 sha = `d75e5c58d01167379c4d9339c628e3969b0385b7a05c9f455b859e4246fad9d4` = 远端原值。且 AST 可达性分析确认：从 `train_rung.py` 出发可达 24 个模块，**`w1_dzero` 不在其中**（`s2_sibling`/`p0_oracles`/`p2b_task` 也都不可达） |
| ⚠️ **意外 3** | 远端 `git status` 说 `scale1m/hf_crawl.py` 被改过，本地没有 | `git diff` 是 **751 增 / 751 删**（全文件 751 行）= 纯换行符churn；两边**工作副本字节完全相同**（都在那 105 个里） |

**⇒ 结论：T6 训练路径上的每一个文件，要么与本地逐字节相同，要么是本篇跑完之后才改的。as-run 对账通过。**
两条意外都是同一个 CRLF/LF 病灶（§2.4）的新病例，坐实了 `.gitattributes` 那个修复的必要性。

> 副产品：远端 `origin` 其实是一个**本地 bundle 文件**（`codes/./r2-100k.bundle`），不是网络仓库。
> 所以「同步代码」= 生成 bundle → `scp` → 从 bundle `fetch`，全程在自己账号内，不涉及任何对外推送。

**同步过程（全部可逆）**：
1. 先把远端整棵源码树打包备份：`~/t6_asrun_source_20260813T164558Z.tgz`（21,468,142 字节，
   sha `719e1e802957ecf9e3cc4aa905c55d113fed60e88bd01473a0b31e5a0a6d5b9c`）
2. 本地增量 bundle `b512213..r2-100k` → `t6more.bundle`（331,389 字节，
   sha `1e1db4ed77044b2fed3b35b8a675fe3c1243f4320f03e12ca7063149b6faea14`，传输后远端复算一致）
3. 删除会被合并覆盖的 untracked 副本（备份已在第 1 步），`git checkout -- .` 清掉换行符改动
4. `git merge --ff-only` ⇒ **`b512213` → `304e11b`**，71 files changed，工作树干净（只剩 `logs/` 等生成物）

### 3.3 P1-3b　vocab 已产出 ✅ ／ 12K 两个 seed 已提交 🟡 排队中

**vocab（作业 `1510203`，`COMPLETED`）**——因为 `$HOME` 是 noexec，dump 必须在计算节点跑，
所以走了一个 4 CPU / 32 G 的小作业，env 从 `/dev/shm` 暂存：

```text
12k  graph 00ac2434…  rows 341  sha256 d4e6b6823edcc5b76f797479ceaf27aec0ead552cac18bcc5f60012b0935ec54
30k  graph e6ae2dfe…  rows 341  sha256 d4e6b6823edcc5b76f797479ceaf27aec0ead552cac18bcc5f60012b0935ec54
```

🔑 **远端（Linux）产出的两份文件与本地（Windows）产出的 sha 逐位相同**，均为 3,686 字节。
这验证了 [`dump_family_vocab.py`](../../scale1m/dump_family_vocab.py) 里显式写 LF 的那行——
跨平台产出同一个指纹，正是 checkpoint 绑定需要的性质。

**两个训练作业**：

| run | JobID | 状态 |
|---|---|---|
| `R2_12k_s1_e25` | **1510204** | PENDING（`Reason=Priority`） |
| `R2_12k_s2_e25` | **1510205** | PENDING（`Reason=Priority`） |

提交时带了 `--dependency=afterok:1510203` 和
`T6_RUNTIME_ENV=/u801/x98liu/model_lake/t6_workflows/T6_20260811T174245Z/runtime.env`。

> 🔴 **为什么必须带 `T6_RUNTIME_ENV`**：那份持久环境里有 `pyg_lib 0.8.0+pt212cu130`，
> 而 `~/.mlf_env` 默认的 `ENV_SOURCE` 里**没有**。少了它，`make_link_loader` 会退回纯 Python 的
> `LightLinkLoader` ⇒ **换了采样器**，seed 1/2 与 seed 0 就不是同一个系统，噪声带白测。

**为什么在排队**：`train_rung.sbatch` 钉了 `--nodelist=watgpu808`，而该节点两张 GPU 当前满载
（`AllocTRES=gres/gpu=2` / `CfgTRES=gres/gpu=2`），要等别人的作业让出来。

> 🔑 **顺带修掉一个会反复咬人的问题：`--time` 开太大，代价不是零。**
> Slurm 按**申请的**墙钟排队，`--time=04:00:00` 让六个**两分钟**的作业在一张 GPU 后面被排成
> **每 4 小时一个**，最后一个排到了 `2026-08-14T13:17`。
> `100kplan.md` §4.5 本来就写了「先用 `--epochs 1` 实测单 epoch，再按实测修正」——现在有实测了
> （最慢 2 分 44 秒，含环境 staging），于是：
> ① 对已排队的六个作业 `scontrol update TimeLimit=00:20:00`
> ⇒ 预约起跑立刻从「跨到明天 13:17」压缩成 **今天 21:16–22:57，每 20 分钟一个**，省掉约 14 小时；
> ② `train_rung.sbatch` 的默认值从 `04:00:00` 改成 `00:30:00`，并写明 R3/R4 必须在命令行上抬高。

**我没有解开这个 pin，这是有意的**：seed 0 的 12K 跑在 watgpu808（H200 NVL）。
换一张卡去跑 seed 1/2，浮点结果会带上硬件差异，而这三个 run 的**全部意义**就是隔离出划分噪声——
把硬件噪声混进去，测出来的带宽就不是我们要的那个。作业只有 93 秒，排队 2.5 小时不浪费任何算力。

**跑完要检查**（收集命令见 §6）：

- [ ] `mechanism_gate.passed == true`（两个 run）
- [ ] 🔑 `metadata.git_head == 304e11b85d7cfff69cfdbf9de3a852b32ea4c780` **非空** —— §2.2 那个修复的运行时验收
- [ ] `metadata.git_status` 为空（树干净）⇒ 不应生成 `uncommitted.patch`
- [ ] 🔑 `binding.family_vocab_sha256 == d4e6b682…` —— P2-1 的运行时验收
- [ ] 新字段 `checkpoints_written` / `checkpoints_retained` / `run_dir_reused: false` 都在
- [ ] 记下两个 run 的 `train_row.head.n_datasets`（seed 0 是 517）—— 12K 档划分噪声的第一组实测

### 3.4 §1.8 裁定 B 的执行：100K 图已带关系类型重建　✅ 图已验 ／ 🟡 run 排队中

**① 补传缺失的输入**（这是整件事的根因）：

```text
36a557958dc464d43a9b28f96d69ec1a9525bb9535566c003a424e28c0d531e5  hf_canon.parquet  (30,487,319 B)
```
传输后远端复算一致；`CANON_REPORT.json` 一并上传。

**② 先保住旧图再重建**——`T6_20260811T174245Z` 那套证据引用的是 `7fbc3c47…`，
文件不能就这么消失，否则那份冻结证据永远无法复验：

```text
hgraph_100k_norel_7fbc3c47.pt        旧图（223,352,080 B，sha 7fbc3c47…）
norel_GRAPH_REPORT_100k.json         旧图的三份报告，同样留存
norel_T5_GATES_100k.json
norel_lineage_stats.json
```

**③ 重建**（作业 `1510210`，RUN_ID `T5_100k_rel_20260813T170032Z`，CPU 4 核，`COMPLETED 0:0`，**用时 1 分 00 秒**）。

**④ 验收——这是一次教科书式的受控改动，只有一个量变了**：

| 量 | 旧图 | 新图 | |
|---|---|---|---|
| `graph_sha256` | `7fbc3c47ca66227c…` | **`16f3521482a8efd69d491b66f84624d9e0086f3cb42288d69a9c7f022492060f`** | 变（预期） |
| `lineage.relation_counts` | `unknown 16385` | **`finetune 10226 / quantized 4360 / adapter 1608 / merge 76 / unknown 115`** | 变（目的） |
| `built_at` | 2026-08-10T21:44:40Z | 2026-08-13T17:01:41Z | 变（预期） |
| `core_graph_sha256` | `e6ae2dfe…` | **相同** | ✅ |
| `ladder_sha256` | `ccf288ae…` | **相同** | ✅ |
| `x_m_sha256` | `d7573a02…` | **相同** | ✅ |
| `family_vocab_sha256` | `f40af358…` | **相同** | ✅ |
| 血缘拓扑 12 项 | — | **逐项相同**（16427 = 42+606+0+15779、components 3709、largest 159、unresolved 10887、self_loop_dropped 400 …） | ✅ |
| 五种边的边数 | — | **逐项相同** | ✅ |
| 文件大小 | 223,352,080 B | **223,352,080 B** | ✅ |
| T5 门禁 | 33/33 | **33/33，`passed: true`** | ✅ |
| test 边 sha | `66c81823…` | **相同**，65,968 边 | ✅ |

🔑 **新图的 `relation_counts` 与本地排练件逐项相同**（`finetune 10226 / quantized 4360 / adapter 1608 / merge 76 / unknown 115`）
⇒ 关系解析是确定性的、可复现的，这次拿到的就是本该在 8-10 号那次拿到的东西。

**⑤ 四个 100K run 已提交**（`RUN_ID` 换了前缀 `R2rel_`，这样 8-11 号那套 run 目录原封不动、两套证据可并排复验）：

| run | JobID | epochs |
|---|---|---|
| `R2rel_100k_s0_e2` | 1510211 | 2（冒烟） |
| `R2rel_100k_s0_e25` | 1510212 | 25 |
| `R2rel_100k_s1_e25` | 1510213 | 25 |
| `R2rel_100k_s2_e25` | 1510214 | 25 |

> **预期这四个 run 的 loss 与 8-11 号那四个几乎相同。** 因为 `relation_weights_applied=false`、
> `weighted_relations=[]`，`relation_id` 目前**不进入任何一项计算**。所以重跑的真正目的有两个：
> ① **绑定完整性**——`hgraph_100k.pt` 这个路径上现在是新图，checkpoint 必须绑新 sha，
> 否则 T7 一加载就会撞 `IncompatibleCheckpoint`；② 让 `r_mm'` 从此有输入可用。
> **而且「loss 应当几乎相同」本身就是一条检查**：如果差得多，说明这次重建不止改了 relation，要查。

---

## 4. 交给 T7/T8 的观察项（现在不下结论）

**① `mean_cos` 随 N 单调上升。** 0.525(12K) → 0.548(30K) → **0.729(100K)**；100K 三个 seed 分别是 0.729 / 0.712 / 0.767。
这是 CLAUDE.md 三大风险之三（hub 附近过平滑）抬头的直接读数，方向与 `100kplan.md` §13 预告的
「`mradermacher` 独占 HALO 的 20.1%」一致。第一个会撞上它的是 **G-D4（HNSW `recall@50 ≥ 0.99`）**。

- T7 的 `tune_ef_for_recall()` 二分上界一开始就放宽，别用 12K 的经验值
- 若 ef 调不上去，按 §13 走「加强同 hub 负样本 / 减一层」，**不要先怀疑索引参数**

**② `head.hit@10` 三档几乎不动。** 12K 0.766 / 30K 0.753 / 100K 0.770。
**这不是 `gold@10`，不能据此下任何判断**——但它正是 §13 表里「`gold@10` 几乎没掉」那一行的形状。
T8 的 **G-D1（候选池真的是 100000）** 要当第一嫌疑查，铁律 3 的断言必须真的执行到。

**③ 三档的 eval 查询集不同**（12K 517 / 30K 734 / 100K seed0 734 / s1 775 / s2 936）。
**G-D5 的单调性 `gold@10(100K) ≤ gold@10(30K) ≤ gold@10(12K)` 必须在统一查询口径下比**，
否则「下降」可能只是查询集换了。30K 与 100K seed0 恰好都是 734，是可直接比的一对；12K 的 517 不是。

**④ 12k/30k 的 checkpoint 导出时记得带上新生成的 vocab**（§1.3 的 `d4e6b682…`），
并在导出件的 binding 里回写 sha。

---

## 5. 出闸门 G-T6b

T7 开工前逐条勾。前四条不过就别开 T7。

- [x] **G-T6b-1** ✅ `hgraph_100k.pt` sha == `7fbc3c47…` 远端实测；T5 门禁 **33/33 全过**；三份报告已回收　→ §3.1
      　　（剩一件文书工作：把 `7fbc3c47…` / `d7573a02…` 回填进 [`T5.md`](T5.md) §2/§8，并改掉「只有两处不同」那句）
- [x] **G-T6b-2** ✅ as-run 对账完成：105 文件逐字节相同，差异全部归因且无一落在训练路径上　→ §3.2
- [ ] **G-T6b-3** 🟡 12K 三个 split seed —— seed 1/2 已提交，预约 `19:28:55` 起跑　→ §3.3
- [ ] **G-T6b-4** 🔴 G-C2 的判据句改写为含 seed 语义的版本（**噪声带随 G-T6b-3 first 次测出**，§1.7）
- [x] **G-T6b-5** `_GIT_ROOT` 修复已完成并有回归测试；**运行时验证**随 §3.3 的新 run 一起完成
- [x] **G-T6b-6** P2-1..P2-5 五条全部完成（§2.4–§2.7），另修一条 CRLF 指纹污染
- [x] **G-T6b-7** P3 的四条观察项已写入 §4，待抄进 T7/T8 开工清单
- [x] **G-T6b-8** ✅ §1.8 已裁定走 **B**：图已带关系类型重建（`16f35214…`，33/33 门过，只有 relation 变）　→ §3.4
- [ ] **G-T6b-9** 🟡 **新增**：四个 `R2rel_100k_*` run 跑完、机制门过、`binding.graph_sha256 == 16f35214…`；
      并与 8-11 号那四个的 loss 对拍（应几乎相同，差得多就要查）　→ §3.4 ⑤

> **G-C2 本身不在这张表里。** 它要等 T7 导出 + T8 评测，`gold10_validated=false` 现在保持正确。
> 本篇做的是**把 G-C2 判得动的前提条件补齐**——图可验、代码可追、seed 口径可比、噪声带有得测。

---

## 6. 产物回收

### 6.1 已回收（`D:\research\model_lake\data\runs\T6more_20260813\`）

| 文件 | sha256 | 来源 |
|---|---|---|
| `t5_remote/GRAPH_REPORT_100k.json` | `1450a4d5cc40c258a120ff5bbd704af43c268b7ddfd1bd51c077135fe15ea591` | §3.1 |
| `t5_remote/T5_GATES_100k.json` | `b3bc1a42f680aeca5f8a1e27480b0018c1b65c7d424683d71e625d7014ab8b86` | §3.1 |
| `t5_remote/lineage_stats.json` | `74730705f960f0917e9b7e51368f4f191cafac56d48a03c16b774c6ce75cdfc7` | §3.1 |
| `remote_code_asrun.sha256` | `e241998d8f9aa8e638205483d6c0452e34bb29900a0656e8355b1b6531850ec5` | §3.2（111 文件） |
| `family_vocab_core_341.csv` | `d4e6b6823edcc5b76f797479ceaf27aec0ead552cac18bcc5f60012b0935ec54` | §3.3（远端产物） |

### 6.2 待作业跑完后回收（预约 `19:28:55` 起跑）

**排队中的六个作业**（全部等 watgpu808 的 GPU；节点 pin 是有意保留的，见 §3.3）：

| JobID | run | 用途 |
|---|---|---|
| 1510204 | `R2_12k_s1_e25` | 12K 划分噪声（P1-3b） |
| 1510205 | `R2_12k_s2_e25` | 同上 |
| 1510211 | `R2rel_100k_s0_e2` | 新图冒烟（§3.4） |
| 1510212 | `R2rel_100k_s0_e25` | 新图 seed 0 |
| 1510213 | `R2rel_100k_s1_e25` | 新图 seed 1 |
| 1510214 | `R2rel_100k_s2_e25` | 新图 seed 2 |

```bash
# 远端：打小证据包（checkpoint 不下载）
source ~/.mlf_env
for r in R2_12k_s1_e25 R2_12k_s2_e25 \
         R2rel_100k_s0_e2 R2rel_100k_s0_e25 R2rel_100k_s1_e25 R2rel_100k_s2_e25; do
  [ -f "$OUTPUT_ROOT/runs/$r/MANIFEST.json" ] || { echo "MISSING $r"; continue; }
  ( cd "$OUTPUT_ROOT/runs/$r" && tar czf ~/t6more_$r.tgz \
      MANIFEST.json metrics metadata stdout/train.log )
done
sacct -j 1510203,1510204,1510205,1510210,1510211,1510212,1510213,1510214 \
  --format=JobID,JobName%22,Submit,Start,End,Elapsed,MaxRSS,AllocTRES,ExitCode
```

```powershell
# 本地：拉回来
scp x98liu@watgpu.cs.uwaterloo.ca:~/t6more_*.tgz `
    D:\research\model_lake\data\runs\T6more_20260813\
```

**回收后逐项核对**（前两条是本篇两个修复的运行时验收，第三条是 §3.4 的对拍）：

1. `metadata.git_head == 304e11b85d7cfff69cfdbf9de3a852b32ea4c780`，`git_status` 为空
2. 12K 两个 run 的 `binding.family_vocab_sha256 == d4e6b682…`；四个 100K run 的 `== f40af358…`
3. 四个 `R2rel_100k_*` 的 `binding.graph_sha256 == 16f35214…`，且 loss 与 8-11 号对应 run 几乎相同

然后跑一遍门禁复验（会顺带把 override 字段写进报告）：

```powershell
.\.venv\Scripts\python.exe scale1m\validate_t6_run.py `
  --run-dir <解包后的 run 目录> --stage 12k --expect-n 12000 --expect-epochs 25 `
  --train-job-id 1510204 --gold-gate-override 0 --report <report.json>
```

checkpoint 照旧不下载。

---

## 7. 明确不做的

- **不重建任何图。** §3.1 果然只验不建（三份报告本来就在远端，连 `verify_rung_graph` 都不必重跑）。
  ⚠️ 这一条现在有一个**待裁定的例外**：§1.8 的 B 方案就是重建。若选 B，本条作废并另行登记。
- **不改任何超参。** T6 的价值一半来自「配置一个字没改」，不开这个口子。
- **不实现 `--amp`。** D-42，推到 R3/R4。
- **不补 30K 的 seed 1/2。** G-D5 的跨档比较用 seed 0 就够。
- **不动已冻结的 evidence 包。** `T6_20260811T174245Z` 是冻结件，更正写在本篇和 `R2_EXECUTION.md`，
  不追改历史产物——这条纪律和 T2 把 v1 `raw/` 降级为审计件是同一条。
- **不替 T4/T5 补登记。** `D-39` / `D-40`（T4.md §306-307）与 `D-41`（T5.md §272）至今**没有进 §16**，
  这是 T4/T5 的收尾欠账，本篇只做记录、不越界代办。
