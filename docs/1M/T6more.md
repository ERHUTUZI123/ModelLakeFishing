# T6 收尾：进 T7 之前要闭合的六件事

上级 runbook：[`100kplan.md`](100kplan.md) §9 / §12。上游执行记录：[`T6GPU.md`](T6GPU.md) §12（工作流 `T6_20260811T174245Z`）。

**一句话：训练本身合格，六个 run 的机制门都是真的过了；卡住 T7 的不是训练，是三条证据链。**

本篇只做一件事——把「跑完了」和「可以往下走」之间的差距列成可执行项。
不新增科学内容，不改任何超参。

---

## 0. 现在到底验到了什么

以下是复核 evidence 包之后，**已经可以当结论用**的部分：

| 结论 | 证据 |
|---|---|
| 四步顺序、依赖链、seeds 并行 | `jobs.tsv` + `sacct_all_jobs.txt` 时间戳 |
| `pyg-lib 0.8.0+pt212cu130` 装上、`WITH_PYG_LIB=True` ⇒ **D-9 解除** | `reports/pyg_probe.txt` + probe 日志里 False→True 的前后对照 |
| 四个 T0 开关六个 run 全对；**配置一个字没改** | 六份 `resolved_config`，与 `l1l3b_config()` 逐项一致 |
| 机制门（G-C1）六个全过；`n_models` 12000/30183/100000 全对 | 六份 `MANIFEST.json` |
| 交付包完整 | 本地重算 `delivery.tgz` 与 6 个 artifact 的 sha256，全部 MATCH |
| 🔴 **铁律 1 在 12K/30K 上是可验的真** | `hgraph_12k.pt` 的 `00ac2434…` 与本地 `hgraph_ml_v2_sub.pt` **逐字节相同**；`hgraph_30k.pt` 的 `e6ae2dfe…` 与 `hgraph_ml_v2.pt` **逐字节相同** |
| checkpoint 契约 | `scale1m/tests/test_checkpoint.py` 本地重跑 18 passed |

**顺带修正 [`T6GPU.md`](T6GPU.md) §6 的一句话。** 那里写「100K 不是简单乘 8.3——L1 的全湖 logQ 项每步都要对全部模型做一次前向，这一项随 N 线性涨」。
实测不是这样，而且证据就在日志里：

```
12k   [lake logQ] models=12000  labeled=11475 max_deg=548
30k   [lake logQ] models=30183  labeled=26508 max_deg=684
100k  [lake logQ] models=100000 labeled=26508 max_deg=684    ← labeled/max_deg 与 30K 完全相同
```

监督集是冻结的 CORE，三档不变；lake logQ 采的是 256 个负例，不是全湖前向。
所以 100K/25ep 的 108.5 s 只比 30K 的 105.5 s 多 3%——**这正是 `100kplan.md` §9.2「代价增量不来自监督量」的实测确认**，
是「监督冻结」设计的一个可报的好处，不是异常。§6 那段预警应改写。

---

## 1. P1：必须闭合，否则 T7 的数字没有落脚点

### 🔴 P1-1　训练用的那张 100K 图，没有 T5 门禁留痕

**问题。** T6 六个 run 绑定的是 `graph_sha256 = 7fbc3c47ca66227c…`。
而 [`T5.md`](T5.md) §2 里记的是 `334153aa…`——那是**本地排练件**（`x_m_sha256 = 6e8b6924…`，对照特征），
T5.md §9 至今写着「watGPU 正式件待跑」。§8 要求跑完发回的三份报告
（`GRAPH_REPORT_100k.json` / `lineage_stats.json` / `T5_GATES_100k.json`）**没有回收**，
本地 `data1m/graphs/` 下那两份是排练版。

后果不是形式问题：[`T6GPU.md`](T6GPU.md) §6 的判卷表有一行「`binding.graph_sha256` 和 T5 报告里的 `graph_sha256` 一致」，
**这一行现在核对不了**；连带 G-B1（铁律 1 六条断言）、G-B2（`check_stage2_contract`）、G-B3（test 边 sha）
在**实际训练用的那份图**上没有任何留痕。

现有的间接旁证是正面的，但都不是门：`binding.num_families = 1896` 与 T5 对账表一致，`n_models = 100000`。

**怎么做。** 在 watGPU 上，先看三份报告在不在（[`build_graph_rung.sbatch`](../../scripts/watgpu/build_graph_rung.sbatch) 会写进 `$DATA_ROOT/data1m/graphs/`）：

```bash
source ~/.mlf_env
ls -la "$DATA_ROOT/data1m/graphs/"
sha256sum "$DATA_ROOT/data1m/graphs/hgraph_100k.pt"     # 必须是 7fbc3c47ca66227c…
```

在的话直接 scp 回来。**不在的话，就地重验，走这条**：

```bash
cd "$PROJECT_ROOT"
srun --cpus-per-task=4 --mem=32G --time=00:30:00 \
  python -m scale1m.verify_rung_graph \
    --rung 100k \
    --graph "$DATA_ROOT/data1m/graphs/hgraph_100k.pt" \
    --core  "$PROJECT_ROOT/stage1BuildTransferGraph/hgraph_ml_v2.pt" \
    --out   "$DATA_ROOT/data1m/graphs/T5_GATES_100k_verify.json"
```

[`verify_rung_graph.py`](../../scale1m/verify_rung_graph.py) 是**就地验**（吃 `--graph`，不重建），
覆盖的正是 G-B1/G-B2/G-B3 三条，包括 `test_edge_sha` 与 CORE 的对拍。

> 🔴 **绝对不许跑 `build_graph_rung`。** 重建会产生一张新的 `hgraph_100k.pt`，
> 六个 checkpoint 的 `binding.graph_sha256` 当场全部作废，T6 要重跑。
> 这一步只验、不建；上面的命令里没有 `--out` 指向 `hgraph_*.pt`，是故意的。

**验收。**

- [ ] `sha256sum` 输出 == `7fbc3c47ca66227c408a78e03197602f0e405bd15a2d7da77fdd10a2f8a651a0`
- [ ] `T5_GATES_100k*.json` 的 `passed == true`，33 条逐条为 true
- [ ] test 边 sha == `66c81823cb731063…`（T5.md §8 的对账值）
- [ ] `edges`：`trained_on 312986 / similar_to 192060 / is_base_of 16427`
- [ ] `family_vocab 1896`（与六份 MANIFEST 的 `binding.num_families` 一致）
- [ ] 报告回收到 `docs/1M/T5_runs/remote/`，**并把 `7fbc3c47…` 与远端 `x_m_sha256` 回填进 T5.md §2/§8**

---

### 🔴 P1-2　代码状态完全没被钉住

**问题。** 六个 run 的 `metadata.git_head` 和 `git_status` **全是空字符串**。
根因查到了，是一行路径：

[`train_rung.py:39`](../../scale1m/train_rung.py#L39) 把 `_REPO_ROOT` 算成 `scale1m/` 的**祖父**目录，
也就是 `codes/`；而 git 仓库根其实是 `codes/ModelLakeFishing/`。
`codes/.git` 是个空壳目录，git 不认，于是 [`_git()`](../../scale1m/train_rung.py#L99) 每次都返回空串。

这不只是少了个字段。因为 `git_status` 返回空，[`train_rung.py:89`](../../scale1m/train_rung.py#L89)
那段「工作树脏就写 `uncommitted.patch`」的分支**从来没触发过**。而工作树其实是脏的：

```
HEAD = b512213 "T4: stage Python environment on node-local filesystem"
 M scale/export_ours.py   scale/global_metrics.py
 M stage2TrainGraphSAGE/{ablation,losses,sampling,top1_eval,train}.py    ← T0 的七项改造
 ?? scale1m/{train_rung,checkpoint,validate_t6_run,gates,…}.py           ← T6 的全部代码
 ?? scripts/watgpu/*.sbatch  scripts/watgpu/submit_t6_all.sh
```

**产出 T6 那六个数字的代码，一行都没有进版本库。** `100kplan.md` §4.3 要求
「生产作业通常使用 clean commit；有意保留的未提交变更写入 `metadata/uncommitted.patch`」——两条都没落地，
G-W3 实际上没过。

**怎么做。** 三步，顺序不能换。

**① 先确认本地那份就是远端跑的那份**（否则提交的 commit 名不副实）：

```bash
# 远端
cd "$PROJECT_ROOT" && find scale1m scripts/watgpu stage2TrainGraphSAGE scale \
  -name '*.py' -o -name '*.sbatch' -o -name '*.sh' | sort | xargs sha256sum > /tmp/remote_code.sha256
```

```powershell
# 本地，同样的清单同样的顺序，逐行比对
scp x98liu@watgpu.cs.uwaterloo.ca:/tmp/remote_code.sha256 $env:TEMP\remote_code.sha256
```

有差异就先停下来查是哪个文件、什么时候改的——**不许直接提交一个和远端不同的树然后叫它 "T6 as-run"**。

**② 提交，commit message 里写清楚这是事后补钉**：

```bash
cd D:/research/model_lake/codes/ModelLakeFishing
git add -A
git commit   # message 见下
```

```
T0+T6: pin the tree that produced run T6_20260811T174245Z

These files were untracked/uncommitted when the six T6 runs executed on
watgpu808 on 2026-08-11; metadata.git_head was empty in all six MANIFESTs
because _REPO_ROOT pointed one level above the repo. Verified byte-identical
to the remote copy before committing (see docs/1M/T6more.md §1 P1-2).
```

**③ 修 `_git()` 的 cwd**，这样下一次运行才会真的记上 commit：

```python
# train_rung.py:39 附近。_REPO_ROOT 不能动 —— sys.path 靠它 import ModelLakeFishing.*
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
_GIT_ROOT  = os.path.dirname(_HERE)      # 新增：仓库根就是 ModelLakeFishing/

# _git() 里把 cwd 换成 _GIT_ROOT
```

**验收。**

- [ ] 远端/本地代码 sha256 清单逐行相同（有差异则先归因，写进 §3 的留痕）
- [ ] commit 已建，hash 记进本篇 §4 的表
- [ ] `_GIT_ROOT` 改完后，P1-3 的新 run 里 `git_head` 非空、`git_status` 为空
- [ ] 加一条单测：`git_head` 在仓库内运行时非空（防回归）

---

### 🔴 P1-3　seed 的语义没定，且 12K 只有一个 seed ⇒ G-C2 判不了

**问题一：`--seed` 是 split_seed，`init_seed` 恒为 0。**

[`train_rung.py:223`](../../scale1m/train_rung.py#L223) 把 `--seed` 喂给 `make_root_aware_splits(split_seed=…)`，
而 [`:262`](../../scale1m/train_rung.py#L262) 的 `init_seed=INIT_SEED` 是个常量。
这与 [`export_ours.py`](../../scale/export_ours.py#L219) 完全一致——**是忠实复用，不是 bug**。
但后果必须在 T7 之前定死口径：三个 seed 换的是**划分**，不是初始化，所以三个 run 的 held-out 测试集不是同一个：

| run | held-out 数据集数 | tau_macro |
|---|---:|---:|
| `R2_100k_s0_e25` | 734 | 0.180 |
| `R2_100k_s1_e25` | 775 | 0.077 |
| `R2_100k_s2_e25` | 936 | 0.217 |

tau 的散布有 2.8 倍。这不是坏事——划分噪声区间比初始化噪声区间更能说明问题——
但**报出去的时候必须写明是哪一种**，否则读者会默认是后者。

**问题二：12K 只跑了一个 seed。**
G-C2 的判据是 `gold@10(R0) = 0.4159 ± 3-seed 噪声`，现在 12K 只有 seed 0，
`MANIFEST.anchor_gold10: 0.4159` 只是回写的常数、不是实测。
而且那条历史噪声带（P3/P4 留下的）是按哪种 seed 算的，现在没查证——
**拿两把不同的尺子量同一个锚点，比不量更糟。**

**怎么做。**

```bash
# 12K 单次 93 秒，补两个 split seed，和 100K 的区间口径对齐
for s in 1 2; do
  RUNG=12k SEED=$s EPOCHS=25 sbatch --export=ALL,RUNG,SEED,EPOCHS \
    scripts/watgpu/train_rung.sbatch
done
```

同时在本地查清历史 `0.4159` 的 seed 语义（`docs/scale/P3/` 与 `P4/`），把结论写成一句话：

> **口径（待填）**：R0 锚点 `0.4159` 是在 split_seed = ___、init_seed = ___ 下测的；
> T6/T7 的 3-seed 区间是 **split 噪声**；两者可比 / 不可比（二选一，给理由）。

**为什么现在跑而不是 T7 之后再说。** 因为 T7 是「导出 + 建索引」，它按 run 目录逐个吃 checkpoint。
现在补齐，T7 一次性导出三份；等 T7 跑完再回头补，等于把 T7 重排一遍。

**验收。**

- [ ] `R2_12k_s1_e25` / `R2_12k_s2_e25` 机制门通过，`git_head` 非空（顺带验证 P1-2 ③）
- [ ] 历史 `0.4159` 的 seed 语义写成一句话，进 [`T6GPU.md`](T6GPU.md) §8 问题 2 那张表
- [ ] G-C2 的判据句被改写成含 seed 语义的版本

---

## 2. P2：不阻塞 T7，但要在 T7 之前补进记录

### P2-1　12k/30k 的 checkpoint 没绑 `family_vocab.csv`

`binding.family_vocab_path` / `family_vocab_sha256` 在这两档都是 `null`，`ckpt/` 里也没有那个文件
（100K 三个 seed 都有，sha `f40af358d64d…`）。CLAUDE.md Step 6 与 `100kplan.md` §9.3 是硬要求。
[`T6GPU.md`](T6GPU.md) §12 那句「100K 三个完整 seed 的 checkpoint 均绑定」措辞是准确的——它绕开了这两档。

缓解事实：这两档的 341 行 vocab 内含在冻结图的 xm0 meta 里，`graph_sha256` 已绑定且**已本地验过字节相同**，
所以行身份仍然可追，没有孤儿风险。

**做法（推荐，不重训）**：T7 导出这两档时，从 `xm0["family_vocab"]` 直接 dump 成 `ckpt/family_vocab.csv`
并把 sha 回写进导出件的 binding。
之所以事后补是正当的：vocab 是图的确定性函数，而图的 sha 已经独立验过——
这与事后补一个**指标**性质完全不同，后者不许。

- [ ] 12k/30k 各产出一份 341 行 `family_vocab.csv`，sha 记进 T7 的 MANIFEST
- [ ] 补一条断言：dump 出来的 vocab 行数 == `binding.num_families`

### P2-2　gold gate override 没写进 gate report 和 SUMMARY

[`T6GPU.md`](T6GPU.md) §11 写「这个 opt-in 会写入工作流的 `jobs.tsv`、每个 gate report 和最终 `SUMMARY.json`」。
实测只写进了 `jobs.tsv` 和 `workflow.env`；六份 gate report 和 `SUMMARY.json` 里**没有** `gold_gate_override` 字段
（只有 `gold10_validated:false` 和一句 note）。不算误导，但文档描述与产物不符。

- [ ] 二选一并执行：在 [`t6_gate_and_advance.sbatch`](../../scripts/watgpu/t6_gate_and_advance.sbatch) 与
      [`t6_summary.sbatch`](../../scripts/watgpu/t6_summary.sbatch) 里补写该字段；**或**改 §11 的措辞
- [ ] 现有 `SUMMARY.json` 不追改（它是冻结件），差异记在本篇

### P2-3　run 目录复用与一次失败尝试没有登记

`runs/R2_12k_s0_e25/stdout/train.log` 顶部有一段**上一次尝试**的 traceback
（`ModuleNotFoundError: matplotlib`，`srun: task 0: Exited with exit code 1`），
而本次工作流的 `logs/train-12k-1508297.out` 是干净的 ⇒ 该 run 目录被复用、`train.log` 是追加写。
本次 `resume_mode=fresh` / `start_epoch=0`，**数字没有被污染**。

两件事要写进 `R2_EXECUTION.md`（D5 明确要求「含失败过程留痕」）：

- [ ] run 目录复用这件事本身，MANIFEST/SUMMARY 里没有任何登记——建议 `train_rung.py` 在检测到已存在的
      非空 `stdout/train.log` 时，往 MANIFEST 写一个 `run_dir_reused: true`
- [ ] matplotlib 是在 probe 阶段被**自动补装**的（见 `logs/probe-1508294.out`），说明 T1 的环境规格不全，
      **G-W2 该拦没拦到**。这是一条真实的过程教训，值得写

### P2-4　`MANIFEST.checkpoints` 字段语义会误导

它取的是 [`train_rung.py:293`](../../scale1m/train_rung.py#L293) 的 `state["saved"]`，
即「**写过的**」而不是「**还在的**」。`keep=3` 会让 `_prune` 剪掉 epoch 4/9，
delivery inventory 里也确实只剩 `last.pt` / `best.pt`。照现在这样，读报告的人会去远端找不存在的文件。

- [ ] 改名为 `checkpoints_written`，或并列加一个 `checkpoints_retained`

### P2-5　`--amp` 的决策要正式登记

`100kplan.md` §9.1 列了 `--amp`，没实现。[`T6GPU.md`](T6GPU.md) §8 已经把理由说清楚并请示，
运行照跑 = 事实上被接受。别把它留在「待你定」状态。

- [ ] 登记为 D-42（或下一个可用编号）写进 `100kplan.md` §16：
      **AMP 推迟到 R3/R4 显存真的不够时再加，加的时候单独跑一次锚点对拍**

---

## 3. P3：交给 T7/T8 的观察项（现在不下结论）

**① `mean_cos` 随 N 单调上升。** 0.525(12K) → 0.548(30K) → **0.729(100K)**。
这是 CLAUDE.md 三大风险之三（hub 附近过平滑）抬头的直接读数，方向与 `100kplan.md` §13 预告的
「`mradermacher` 独占 HALO 的 20.1%」一致。第一个会撞上它的是 **G-D4（HNSW `recall@50 ≥ 0.99`）**。

- [ ] T7 的 `tune_ef_for_recall()` 二分上界一开始就放宽，别用 12K 的经验值
- [ ] 若 ef 调不上去，按 §13 走「加强同 hub 负样本 / 减一层」，**不要先怀疑索引参数**

**② `head.hit@10` 三档几乎不动。** 12K 0.766 / 30K 0.753 / 100K 0.770。
**这不是 `gold@10`，不能据此下任何判断**——但它正是 §13 表里「`gold@10` 几乎没掉」那一行的形状。
T8 的 **G-D1（候选池真的是 100000）** 要当第一嫌疑查，铁律 3 的断言必须真的执行到。

**③ 三档的 eval 查询集不同**（12K 517 / 30K 734 / 100K seed0 734）。
**G-D5 的单调性 `gold@10(100K) ≤ gold@10(30K) ≤ gold@10(12K)` 必须在统一查询口径下比**，
否则「下降」可能只是查询集换了。30K 与 100K 恰好都是 734，12K 是 517——
T8 要么取交集，要么明确报「三档各自口径」并说明不可直接相减。

---

## 4. 执行顺序

```
P1-2 ①②③ 钉住代码（本地，先做——后面的 run 才有 commit 可记）
      │
      ├─ P1-1 T5 就地重验 + 报告回收（远端，不依赖代码提交，可并行开工）
      │
      └─ P1-3 12K seed 1/2（远端；跑之前先 git pull，验证 git_head 非空）
              │
              └─ P2-1..P2-5 记录补齐（本地）
                      │
                      └─ 出闸门 G-T6b ⇒ 开 T7
```

P1-1 和 P1-2 没有依赖，可以同时开。P1-3 必须排在 P1-2 ③ 之后——
它的第二个作用就是验证 `git_head` 的修复。

| 项 | 在哪跑 | 估计代价 |
|---|---|---|
| P1-1 | watGPU，1 个 CPU 作业 | < 10 min（含排队） |
| P1-2 | 本地 | 30 min（比对 + 提交 + 改一行 + 单测） |
| P1-3 | watGPU，2 个 GPU 作业 | 2 × 93 s + 排队 |
| P2-* | 本地文档与小改 | 1–2 h |

**待填：** P1-2 的 commit hash = `________`

---

## 5. 出闸门 G-T6b

T7 开工前逐条勾。前四条不过就别开 T7。

- [ ] **G-T6b-1** 🔴 `hgraph_100k.pt` 的 sha == `7fbc3c47…`，且其 T5 门禁报告 33 条全过、已回收、已回填 T5.md
- [ ] **G-T6b-2** 🔴 T6 as-run 的 commit 已建，且经远端 sha256 清单确认与远端一致
- [ ] **G-T6b-3** 🔴 12K 三个 split seed 的 checkpoint 齐备，机制门全过
- [ ] **G-T6b-4** 🔴 G-C2 的判据句已改写为含 seed 语义的版本（split 噪声 vs init 噪声写明）
- [ ] **G-T6b-5** `_GIT_ROOT` 修复已验证：新 run 的 `git_head` 非空
- [ ] **G-T6b-6** P2-1..P2-5 五条各自完成或明确记为「已知偏差 + 理由」
- [ ] **G-T6b-7** P3 的三条观察项已抄进 T7/T8 的开工清单

> **G-C2 本身不在这张表里。** 它要等 T7 导出 + T8 评测，`gold10_validated=false` 现在保持正确。
> 本篇要做的是**把 G-C2 判得动的前提条件补齐**——图可验、代码可追、seed 口径可比。

---

## 6. 跑完发我什么

| # | 内容 | 对应 |
|---|---|---|
| 1 | `T5_GATES_100k*.json` + `GRAPH_REPORT_100k.json` + `lineage_stats.json`（远端那份） | P1-1 |
| 2 | `sha256sum $DATA_ROOT/data1m/graphs/hgraph_100k.pt` 的输出 | P1-1 |
| 3 | 远端 `/tmp/remote_code.sha256`，以及与本地的 diff（哪怕是空的） | P1-2 |
| 4 | commit hash + `git log -1` | P1-2 |
| 5 | `R2_12k_s{1,2}_e25` 的 `MANIFEST.json` + `metrics/train_history.json` | P1-3 |
| 6 | 这两个新 run 的 `metadata.git_head`（非空即是修复生效） | P1-2 ③ |

checkpoint 照旧不下载。

---

## 7. 本篇明确不做的

防范围蔓延，写死：

- **不重建任何图。** P1-1 只验不建（理由见该节的红框）。
- **不改任何超参。** T6 的价值一半来自「配置一个字没改」，这里不许开口子。
- **不实现 `--amp`。** 见 P2-5，推到 R3/R4。
- **不补 30K 的 seed 1/2。** G-D5 的跨档比较用 seed 0 就够；30K 的区间没有需求方。
- **不动已冻结的 evidence 包。** `T6_20260811T174245Z` 是冻结件，所有更正写在本篇和 `R2_EXECUTION.md`，
  不追改历史产物——这条纪律和 T2 把 v1 `raw/` 降级为审计件是同一条。
