# v6 S2 可审计记录 —— serving-time sibling-prior 融合(headline,零重训)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 S 轨 / §0.6-3(合法性)
**性质:** 纯 serving 层重排,**不引入任何可训练参数、不改动已上线模型**——
在冻结的 L1L3b 嵌入上加一步分数融合。O-sibling(P0 上界 0.563)的可部署实现。

---

## 1. 机制

查询数据集 D 到达时,除 MIPS 分数 `z_d·z_m` 外,额外用 D 在**湖内**的同根
兄弟数据集(有 trained_on 标签、训练可见)的已知高分模型作先验:

```
boost(m) = mean_{D' ∈ siblings(D), D' 训练可见, D' ≠ D}  norm_acc(m, D')   (无兄弟证据则 0)
fused(D, m) = minmax(z_d·z_m)[m] + α · boost(m)
排全湖 9,491 模型 → gold@K / gold-gap@K
```

**合法性边界(§0.6-3,代码断言):** boost 只用**训练可见的兄弟节点**
(同根、不同数据集节点、边在训练图内),`sib = nodes − {D}` 显式排除 D 自身
——**绝不使用 D 自己的 held-out 标签**(那是 O-sibling oracle 的泄漏部分)。

## 2. 为什么用 node-level 切分而非 root-aware(关键设计)

root-aware 切分把**整根**held-out → 测试数据集的兄弟也全在测试侧 →
**零训练可见兄弟 → S2 无从触发**(这正确地对应 E19 的"全新孤立根"场景,
内容是那里唯一的指望,而内容已证伪)。S2 针对的是**另一个真实服务场景**:
一个新数据集,其兄弟 config 已在湖内(如 MTEB/tatoeba 新增语言 config)。
故用 **node-level 数据集切分**:D 自己的边全部 held-out(对 D 冷),
但 D 的兄弟节点留在训练(有标签)——`cold-D、warm-siblings`。
断言 `train 消息图不含任何 test 节点的边`(对 D 严格冷)。

这与 E19 完美对应:
- root-aware(全新根)→ S2 不触发 → 孤立层,近天花板;
- node-level(已知根的新兄弟)→ S2 触发 → sibling-rich 层。

## 3. 代码改动

| 文件 | 性质 |
|---|---|
| `stage2TrainGraphSAGE/s2_sibling.py` | **新增**:`node_level_dataset_split`(cold-D/warm-sibling,含冷-D 断言);`s2_scores`(minmax MIPS + α·sibling boost,排除 D 自身);8-seed × α 扫描 × has_sibling 分层评测;产 `s2_report.json` / `s2_rows.csv` |

零改动:训练配方(L1L3b)、模型、已上线索引均不动。S2 是评测/服务侧的纯重排。

## 4. 冒烟结果(1 seed / 8 epochs,方向验证)

| 分层 | α | gold@10 | gold-gap@10 | gold@1 |
|---|---|---|---|---|
| **has_sibling**(94 ds) | 0.0(纯 MIPS) | 0.062 | 0.162 | 0.019 |
| **has_sibling** | **0.5(S2)** | **0.253** | **0.408** | **0.119** |
| no_sibling(23 ds) | 0.0 | 0.174 | 0.348 | 0.000 |
| no_sibling | 0.5 / 1.0 / 2.0 | **0.174(不变)** | 0.348(不变) | 0.000 |

**两个判读:**
1. **has_sibling 层 gold@10 0.062 → 0.253(4×)、gold@1 6×**——S2 在兄弟密集的
   数据集上把"同根已知最优"信息直接兑现,接近 O-sibling 头顶;
2. **no_sibling 层跨 α 完全不变**——合法性边界工作:S2 只在有训练可见兄弟处
   触发,无兄弟处 boost≡0,零副作用。这是 S2 与 oracle 的分界的实证。

## 5. 8-seed 终裁(2026-07-19)—— **S2 采纳(sibling-rich 场景)**

root_macro,α=0.0 = 纯 MIPS 基线,α=1.0 = 最优 S2(α=0.5 几乎相同):

| 分层 | n/seed | α | gold@10 | gold-gap@10 | gold@1 |
|---|---|---|---|---|---|
| **has_sibling** | 92 | 0.0(MIPS) | 0.162 | 0.267 | 0.038 |
| **has_sibling** | 92 | **1.0(S2)** | **0.266** | 0.330 | **0.122** |
| no_sibling | 30 | 0.0 | 0.116 | 0.187 | 0.004 |
| no_sibling | 30 | **任意 α** | **0.116(不变)** | 0.187(不变) | 0.004(不变) |
| all | 123 | 0.0 → 1.0 | 0.137 → 0.188 | 0.222 → 0.254 | 0.020 → 0.063 |

**统计裁决(has_sibling,paired,同数据集同 z):**
- **gold@10:8/8 seeds 全升**;flat 配对 Δ = **+0.065,CI [+0.040, +0.089] 排除 0**;
  root_macro **0.162 → 0.266(+64%)**;
- gold@1:0.038 → 0.122(3.2×);
- gold-gap@10:root_macro 0.267 → 0.330(升),flat 配对跨 0(S2 提升的是"精确
  gold"的排名,而 gap 已计入近平局、松弛余量本就小,二者不矛盾);
- **no_sibling 层跨所有 α、所有 8 seed 逐位不变**——合法性边界绝对可靠,
  S2 零副作用。

**这是 v3→v6 全程自 L1 以来第一个干净、显著、8/8 一致的正结果,且零重训。**

**§0.4 sibling-rich 层目标对照:**
- root gold@10 = 0.266 **落入目标带 0.26–0.35 ✓**(关闭现任→O-sibling
  0.10→0.563 差距的 **36%**,达 35–55% 下沿);
- gold-gap@10 = 0.330 未达 ≥0.50(部分)——gap 口径下 headroom 更靠 S3
  (兄弟辅助监督)或更强融合。

**采纳决定:** S2(α=1.0)进入 serving 栈,针对 **sibling-rich 查询**(82% gold
覆盖,E18)。**最优 α=1.0**(gold@10/gold@1 最优);α=0.5 备选(gold-gap@10 更高
0.340,gold@10 几乎相同)。**孤立根(no_sibling)S2 不触发、零副作用**——
两类根共用一套服务逻辑,S2 自动只在有兄弟处生效。

**部署形态:** serving 层在 HNSW top-K 之后加一步——查询根的湖内兄弟 → 取其
高分模型 → 融合重排。索引侧需存 root→dataset 映射 + trained_on 邻接
(轻量 sidecar),Stage-3 export 已有 occupancy/id 快照可扩展。

## 6. 复现

```bash
cd ModelLakeFishing  # 仓库根
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.s2_sibling \
    --seeds 0 1 2 3 4 5 6 7 --epochs 25
# 产物: stage2TrainGraphSAGE/artifacts/ablation/d0/s2_sibling/s2_report.json
```