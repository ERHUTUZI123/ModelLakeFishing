# P3 执行报告：在 ModelLens 语料图上训练 L1L3b 冠军 + 导出 z / HNSW

**执行日期：** 2026-07-27
**对应规划：** `docs/scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md` §3.2 / §3.3、§5 P3 行、决策 D-3
**状态：** ✅ **P3 出闸通过**（mechanism-first smoke 过；HNSW recall@50 = 1.000 > 90%；harness 端到端一致）；**因 8 GiB 显存双重 OOM，按 D-3 子采样到 12K/3K 训练**
**产物：**
- 代码：`scale/modellens_subsample.py`、`scale/export_ours.py`
- 图：`stage1BuildTransferGraph/hgraph_ml_v2_sub.pt`（12,000 模型 / 3,000 数据集）
- 导出：`docs/scale/P3/exports/ml_sub_L1L3b/`（z_m/z_d、id 表、gold 标签、HNSW 索引、报告）

---

## 0. 一句话结论

用**未改动的** L1L3b 冠军配置（同 D0 那套）在 ModelLens 语料图上训练成功，导出 `z_m/z_d` + HNSW 索引 + gold 标签，为 P4 的 A/B 轴备好我方系统。**我方 gold@10 = 0.416 / root_gold@10 = 0.392 / top3@10 = 0.532**（12K 候选、517 个 held-out 测试数据集）；**HNSW recall@50 = 1.000、p50 = 0.069ms**。但 8 GiB 笔记本 GPU 触发两处 O(N²)/O(N_d) OOM，torch 2.12 又无 pyg-lib 轮子，故按 PLAN §3.2 / D-3 **子采样到 12K 模型 / 3K 数据集**（保留全部 1,686 gold 节点）—— 对照仍受控，但"全 47K"规模在此硬件上未达成（见 §4 约束）。

---

## 1. 过程（含两处显存墙，均为过程留痕）

### 1.1 入口 + 冠军驱动
`verify_corpus` 绿。冠军训练直接复用 `stage2TrainGraphSAGE/w1_dzero.py`（`--graph` 参数化），L1L3b = L1 全湖 logQ 采样 softmax + L3 native task + n_neg 256，**代码零改动**（科学诚信：与 D0 夺冠的是同一个 L1L3b）。

### 1.2 🟠 显存墙 1（GPU）：O(N_model²) 对比损失
`hgraph_ml_v2.pt`（30,183 模型）直训即 **CUDA OOM**：`contrastive_loss` 构造 `[N,N]` 相似度/掩码，30,183² × 4 × 4 张 ≈ 11 GB，远超 8.6 GiB。根因是**无 pyg-lib** ⇒ loader 退回 full-neighbour（子图=近全图），`z["model"]` 覆盖全部模型。（代码注释明言 pyg-lib 才有真正的邻域采样 LinkNeighborLoader。）

### 1.3 🟠 显存墙 2（CPU）：O(n_sample × N_dataset) 密度诊断
子采样模型后又撞 **CPU OOM 6.27 GB**：`global_positive_density` 抽 200,000 对 × `N_dataset` 列，7,843 数据集 × 200K × 4 = 6.27 GB，超 10.6 GB 可用内存的安全线。⇒ 数据集轴也须设上限。

### 1.4 D-3 子采样（PLAN §3.2 明文兜底）
`scale/modellens_subsample.py`：确定性选择 —— ① 每 gold 节点的 gold+top3 模型（评测关键，全留）；② 按 trained_on 度降序补到 `--budget`；③ 数据集 = 全 gold 节点 + 按边数补到 `--max-datasets`。**12,000 模型 / 3,000 数据集**（含全部 1,686 gold）/ 281,887 trained_on 边。**两系统在 P4 都限定此宇宙 ⇒ 对照仍受控**。

### 1.5 smoke 机制门（通过）
`--epochs 2 --seeds 0`：loss 下降、指标合理、无 NaN。**gold@10 = 0.294 @2ep** ⇒ 机制正确，放行全量。

### 1.6 训练 + 导出
`scale/export_ours.py`：L1L3b seed 0 / 25 epochs（**980s ≈ 16 分钟**，GPU），over 全图导出 `z_m [12000] / z_d [3000]`，评测 held-out gold@K，建 HNSW。

---

## 2. 结果：我方系统在 ModelLens 子湖上

（`export_report.json`；候选宇宙 12,000 模型，517 个 root-aware held-out 测试数据集 / 412 roots）

| 指标 | flat | root-macro |
|---|---:|---:|
| gold@1 | 0.0948 | 0.0878 |
| **gold@10** | **0.4159** | **0.3920** |
| top3@10 | 0.5319 | 0.5065 |
| gold-gap@10 | 0.5087 | 0.4958 |
| observed hit@1 | 0.358 | — |
| median gold rank（/12000） | **21** | — |

**harness 端到端一致性**：同一次运行里 `top1_eval.five_metric_eval` 与 `scale/global_metrics.from_scores` 的 gold@10 **完全相等（0.41586 == 0.41586, match=True）** —— 证明 P2 造的 harness 不只在合成数据上、在真实训练输出上也与既有评测器口径一致。

**HNSW（B 轴基础 + 保真门）**：
| 项 | 值 |
|---|---:|
| recall@50 vs 暴力 MIPS | **0.9999**（> 90% 门，通过）|
| query p50 / p99 | **0.069ms / 0.196ms** |
| build（12K 建索引） | 228ms |

---

## 3. 与 ModelLens P2 基线的对读（**非最终判决，P4 才是**）

| | 我方 L1L3b（P3） | ModelLens（P2, 盲数据集） |
|---|---:|---:|
| gold@10 | **0.416** | 0.0024 |
| 候选宇宙 | 12,000 | 47,242 |
| median gold rank | 21 / 12,000 | 5,762 / 47,242 |

**诚实边界（必须一起读）**：
1. **宇宙不同**（12K vs 47K），绝对值不可直接比 —— P4 须把两系统放到**同一** 12K 宇宙再比。
2. ModelLens 的 0.0024 是**被未发布件致盲**的 release 版（P2 §3.1），**低估**原作满血系统。**这不是碾压**，是"可复现 release 版"的真实表现。
3. 故此表仅为**方向性对读**；P4 会在同宇宙、带完整披露下出正式 A 轴。

即便如此，两点结构性观察成立且有价值：我方在**同一份 ModelLens 监督**上，用两段式图表示把 dataset-specific 的 gold 顶到中位第 21 名；HNSW 亚线性检索 0.069ms/查询——这正是 B 轴要坐实的护城河的雏形。

---

## 4. 对下一步的影响 + 新约束

### 4.1 P4 已就绪的输入
- 我方 `z_m/z_d` + `hnsw_index.bin` + `gold_cands.npz`（held-out 标签）已导出；
- A 轴：`global_metrics.from_embeddings(z_m, z_d, cands)` 直接出我方数字，与 ModelLens 适配器同 harness；
- B 轴：我方 HNSW p50 0.069ms 已量；ModelLens 的 O(N) 全扫延迟用 P2 适配器 `score_matrix` 量，P4b 对比。

### 4.2 🔴 新约束 D-9：全 47K/30K 规模受阻于硬件 + pyg-lib 缺失
- **8 GiB 笔记本 GPU** 装不下 30K 的 O(N²) 对比损失；**torch 2.12 无 pyg-lib 预编译轮子**（Windows），拿不到邻域采样 loader。
- 现状：**12K 子采样是本机能达到的最大受控规模**（≈1.3× D0 的 9.5K）。"5× 放大到 47K"的叙事在此硬件上**未兑现**。
- **选项（供裁定）**：(a) 换 ≥24 GB GPU / 云机跑全 30K；(b) 装/编译 pyg-lib（换到有轮子的 torch 版本）拿到邻域采样，则 8 GiB 也能训全 30K；(c) 接受 12K 受控口径，P4 明写"受控子采样规模"、B 轴用延迟-vs-N 曲线外推 47K/1M 的渐近结论（曲线斜率不依赖训练规模）。
- **倾向 (c) + 视资源补 (a)**：(c) 已足以支撑受控 head-to-head 与渐近护城河论证；全规模作 stretch。

### 4.3 血缘边（D-8）现状
子采样后 is_base_of 仅 32 条（原 42），与 P1 结论一致 —— C 轴不靠血缘，靠 sibling/task 融合 + inductive，符合 D-8 倾向 (c)。

---

## 5. 可复现命令

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
.\.venv\Scripts\python.exe -m scale.verify_corpus --only v2
# D-3 子采样（12K 模型 / 3K 数据集，保全 gold）
.\.venv\Scripts\python.exe -m scale.modellens_subsample --budget 12000 --max-datasets 3000
# 训练 L1L3b 单 seed + 导出 z + HNSW
Set-Location D:\research\model_lake\codes
ModelLakeFishing\.venv\Scripts\python.exe -m ModelLakeFishing.scale.export_ours `
  --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt --seed 0 --epochs 25 --tag ml_sub_L1L3b
```
