# T6 收尾：进 T7 之前要闭合的事

上级 runbook：[`100kplan.md`](100kplan.md) §9 / §12。上游执行记录：[`T6GPU.md`](T6GPU.md) §12（工作流 `T6_20260811T174245Z`）。

**一句话：训练本身合格，六个 run 的机制门都是真的过了；卡住 T7 的不是训练，是三条证据链。**

> **执行状态（2026-08-13）**：本地能做的 **9 项全部完成并已验证**（§2）；
> 需要 watGPU 的 **3 项仍然阻塞**（§3）——本机 SSH 走不通（`x98liu@watgpu.cs.uwaterloo.ca: Permission denied (publickey)`，
> 私钥按 `100kplan.md` §4.2 设了 passphrase，非交互会话无法解锁）。§3 是一段可直接粘贴的命令。
> 所有已确定的数字与指纹在 **§1 指纹台账**，那一节是本篇给后续阶段的主要交付。

---

## 0. 执行状态总表

| # | 项 | 状态 | 在哪看 |
|---|---|---|---|
| P1-1 | 100K 图的 T5 门禁留痕 | 🔴 **阻塞（需 watGPU）** | §3.1 |
| P1-2 ① | 远端 as-run 代码与本地对账 | 🔴 **阻塞（需 watGPU）** | §3.2 |
| P1-2 ② | 提交 T6 源码树 | ✅ 完成 `3992d67` | §2.1 |
| P1-2 ③ | 修 `_git()` 的 cwd + 回归测试 | ✅ 完成，7 个新测试 | §2.2 |
| P1-3a | seed 语义定案 | ✅ **已查清并登记 D-43** | §2.3 |
| P1-3b | 12K 补 seed 1/2 | 🔴 **阻塞（需 watGPU）** | §3.3 |
| P2-1 | 12k/30k 的 `family_vocab.csv` | ✅ 完成，已生成并验证 | §2.4 |
| P2-2 | override 写进 gate report / SUMMARY | ✅ 完成，端到端验过 | §2.5 |
| P2-3 | run 目录复用登记 | ✅ 完成 | §2.6 |
| P2-4 | `checkpoints` 字段语义 | ✅ 完成 | §2.6 |
| P2-5 | `--amp` 决策登记 | ✅ 完成 **D-42** | §2.7 |
| ＋ | **新发现**：CRLF 污染 vocab 指纹 | ✅ 已修 + 已加 `.gitattributes` | §2.4 |

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
| 100k | `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0` | **只在远端** | 🔴 **未核验（P1-1）** |

> 12k/30k 这两行是**铁律 1 在这两档上的直接证据**：训练用的图与本地冻结的 CORE 图逐字节相同。
> 100k 那一行是本篇的头号阻塞——本地那份 `hgraph_100k.pt` 是**排练件**，不是它。

### 1.2 100K 图：本地排练件 vs 远端正式件

| 量 | 本地排练件（`GRAPH_REPORT_100k.json`，2026-08-10T17:51:36Z） | 远端正式件（T6 实际训练用） |
|---|---|---|
| `graph_sha256` | `334153aac46d705ccab79012bdbd30402b863ac9aedee2752ca69f67d32ea004` | `7fbc3c47ca66227c…` |
| `x_m_sha256` | `6e8b69245a462cd5299a10c49565784c749ab5d23d6314934557948901ba7fd3`（对照特征） | `d7573a02…`（T5.md §8 记的正式特征） |
| `family_vocab_sha256` | `0fbf5c11d3cae2ea8254ee8ac218babe158c0c932a43c4ce002d7a64dc4aa157` 🔴 **CRLF 污染** | `f40af358d64d96a2…`（T6 六个 run 绑定的 LF 形式） |
| `core_graph_sha256` | `e6ae2dfeb59779f4cb0242a08bf90d6628699e90a9a648048b799d9cc7d71cda` | 应相同 |
| `ladder_sha256` | `ccf288aee2a3caee10b60ab8a7ee3031f44d5766e33074e4090eb35cb128b0ab` | 应相同 |

> 🔴 **T5.md §8 的对账清单要改一个字。** 那里写「**只有** `graph_sha256` 和 `provenance.x_m_sha256` 会不同」。
> 实际上 **`family_vocab_sha256` 也会不同**，而且原因与科学无关——纯粹是换行符（§2.4）。
> 不先说清楚，P1-1 回收报告时会看到第三个不一致，然后花时间查一个不存在的问题。

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

## 3. 仍然阻塞的（需要 watGPU）

本机 SSH 不通：`ssh x98liu@watgpu.cs.uwaterloo.ca` → `Permission denied (publickey)`。
`~/.ssh/id_ed25519` 在，但按 `100kplan.md` §4.2 设了 passphrase，非交互会话解不开。
下面三段按顺序粘贴即可，**中间不需要我参与**。

### 3.1 P1-1　100K 图的 T5 门禁留痕　🔴 头号阻塞

```bash
source ~/.mlf_env
cd "$PROJECT_ROOT"

# ① 图本体的指纹必须是这个值
sha256sum "$DATA_ROOT/data1m/graphs/hgraph_100k.pt"
#   期望 7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0

# ② 三份 T5 报告在不在
ls -la "$DATA_ROOT/data1m/graphs/"

# ③ 不在就地重验（不重建！）
srun --cpus-per-task=4 --mem=32G --time=00:30:00 \
  python -m scale1m.verify_rung_graph \
    --rung 100k \
    --graph "$DATA_ROOT/data1m/graphs/hgraph_100k.pt" \
    --core  "$PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
    --out   "$DATA_ROOT/data1m/graphs/T5_GATES_100k_verify.json"
```

> 🔴 **绝对不许跑 `build_graph_rung`。** 重建会产生一张新的 `hgraph_100k.pt`，
> 六个 checkpoint 的 `binding.graph_sha256` 当场全部作废，T6 要重跑。
> [`verify_rung_graph.py`](../../scale1m/verify_rung_graph.py) 吃 `--graph`、**只验不建**，
> 覆盖的正是 G-B1（铁律 1 的字节断言）、G-B2（`check_stage2_contract`）、G-B3（`test_edge_sha` 与 CORE 对拍）。

**对账清单**（`T5.md` §8 的那份，加上本篇查出来的一条修正）：

- [ ] `graph_sha256` == `7fbc3c47ca66227c…`
- [ ] 33 条门逐条 true，`passed == true`
- [ ] test 边 sha == `66c81823cb731063…`
- [ ] `trained_on 312986 / similar_to 192060 / is_base_of 16427`
- [ ] `lineage 16427 = core_core 42 + core_halo 606 + halo_halo 15779`
- [ ] `family_vocab 1896`（与六份 MANIFEST 的 `binding.num_families` 一致）
- [ ] 🔴 **预期会有三处不同，不是两处**：`graph_sha256`（`334153aa` vs `7fbc3c47`）、
      `x_m_sha256`（`6e8b6924` vs `d7573a02`）、**`family_vocab_sha256`（`0fbf5c11` vs `f40af358`，纯 CRLF/LF 之差，见 §2.4）**。
      **第四处不同才是问题。**

### 3.2 P1-2 ①　远端 as-run 代码对账

```bash
cd "$PROJECT_ROOT"
find scale1m scripts/watgpu stage2TrainGraphSAGE scale -type f \
  \( -name '*.py' -o -name '*.sbatch' -o -name '*.sh' \) \
  -not -path '*/__pycache__/*' | sort | xargs sha256sum > /tmp/remote_code.sha256
wc -l /tmp/remote_code.sha256      # 应为 115 行（本地清单是 115 行 + 3 行注释头）
```

拿回本地后与 [`T6more_runs/local_code.sha256`](T6more_runs/local_code.sha256) 比对
（本地清单自身 sha `c482bdaa1908fcbd…`，对应提交 `3992d67`）：

```powershell
scp x98liu@watgpu.cs.uwaterloo.ca:/tmp/remote_code.sha256 $env:TEMP\remote_code.sha256
```

**判读规则（重要）**：差异**应当恰好**是这 9 个文件——它们是 T6 跑完之后本篇改的：

```
scale1m/train_rung.py                       scripts/watgpu/train_rung.sbatch
scale1m/validate_t6_run.py                  scripts/watgpu/t6_gate_and_advance.sbatch
scale1m/dump_family_vocab.py      (新)      scripts/watgpu/t6_summary.sbatch
scale1m/tests/test_run_metadata.py (新)     (docs 不在这个清单里)
```

**多出任何一个差异 = 远端跑的不是本地这份代码，必须查清再进 T7。**

### 3.3 P1-3b　12K 补两个 split seed

```bash
cd "$PROJECT_ROOT"
git pull --ff-only            # 拿到 3992d67 / f013027

# 先把 CORE 的 vocab 落到 sbatch 会找的位置（否则新版脚本会按 §2.4 直接失败）
for r in 12k 30k; do
  case $r in 12k) g=hgraph_ml_v2_sub.pt;; 30k) g=hgraph_ml_v2.pt;; esac
  python -m scale1m.dump_family_vocab \
    --graph "$PROJECT_ROOT/stage1BuildTransferGraph/$g" \
    --out   "$DATA_ROOT/data1m/feats/$r/family_vocab.csv"
done
#   两次都应打印 rows 341 / sha256 d4e6b6823edcc5b7…

for s in 1 2; do
  RUNG=12k SEED=$s EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
    scripts/watgpu/train_rung.sbatch
done
```

单次 93 秒。跑完检查两件事：
`mechanism_gate.passed == true`，以及 **`metadata.git_head` 非空**——后者就是 §2.2 那个修复的验收。

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

- [ ] **G-T6b-1** 🔴 `hgraph_100k.pt` 的 sha == `7fbc3c47…`，且 T5 门禁 33 条全过、报告已回收、已回填 T5.md　→ §3.1
- [ ] **G-T6b-2** 🔴 远端 as-run 代码对账完成，差异恰好是那 9 个文件　→ §3.2
- [ ] **G-T6b-3** 🔴 12K 三个 split seed 的 checkpoint 齐备，机制门全过　→ §3.3
- [ ] **G-T6b-4** 🔴 G-C2 的判据句改写为含 seed 语义的版本（**噪声带须先补测**，§1.7）
- [x] **G-T6b-5** `_GIT_ROOT` 修复已完成并有回归测试；**运行时验证**随 §3.3 的新 run 一起完成
- [x] **G-T6b-6** P2-1..P2-5 五条全部完成（§2.4–§2.7），另修一条 CRLF 指纹污染
- [x] **G-T6b-7** P3 的四条观察项已写入 §4，待抄进 T7/T8 开工清单

> **G-C2 本身不在这张表里。** 它要等 T7 导出 + T8 评测，`gold10_validated=false` 现在保持正确。
> 本篇做的是**把 G-C2 判得动的前提条件补齐**——图可验、代码可追、seed 口径可比、噪声带有得测。

---

## 6. 跑完发我什么

| # | 内容 | 对应 |
|---|---|---|
| 1 | `T5_GATES_100k*.json`（+ 若在，`GRAPH_REPORT_100k.json` / `lineage_stats.json`） | §3.1 |
| 2 | `sha256sum $DATA_ROOT/data1m/graphs/hgraph_100k.pt` 的输出 | §3.1 |
| 3 | `/tmp/remote_code.sha256` | §3.2 |
| 4 | `R2_12k_s{1,2}_e25` 的 `MANIFEST.json` + `metrics/train_history.json` | §3.3 |
| 5 | 这两个新 run 的 `metadata.git_head`（非空即是修复生效） | §3.3 |
| 6 | 两次 `dump_family_vocab` 的输出（应为 `rows 341 / d4e6b682…`） | §3.3 |

checkpoint 照旧不下载。

---

## 7. 明确不做的

- **不重建任何图。** §3.1 只验不建。
- **不改任何超参。** T6 的价值一半来自「配置一个字没改」，不开这个口子。
- **不实现 `--amp`。** D-42，推到 R3/R4。
- **不补 30K 的 seed 1/2。** G-D5 的跨档比较用 seed 0 就够。
- **不动已冻结的 evidence 包。** `T6_20260811T174245Z` 是冻结件，更正写在本篇和 `R2_EXECUTION.md`，
  不追改历史产物——这条纪律和 T2 把 v1 `raw/` 降级为审计件是同一条。
- **不替 T4/T5 补登记。** `D-39` / `D-40`（T4.md §306-307）与 `D-41`（T5.md §272）至今**没有进 §16**，
  这是 T4/T5 的收尾欠账，本篇只做记录、不越界代办。
