# v4 收官可审计记录 —— Stage-3 重索引(L1L3b @ D0)+ W3 卫生行(D2-P2 degree-cap)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 W1-F(重索引)与 W3(卫生轨)
**前提:** L1L3b 已依 root_macro 口径晋升 D0 默认候选(`W1_D0_AUDIT.md` §8)。

---

## 1. Stage-3 重导出/重索引(管线零改动的验证)

### 1.1 流程(`stage3HNSW/d0_freeze_export.py`,新增)

```
FREEZE : hgraph_d0_v1 + root-aware split 0 + 记录的 init seed 重训 L1L3b
         → 功能性冻结门(见 1.2)→ vocab 绑定 checkpoint
EXPORT : export_embeddings.export()(既有代码,零改动)
         → 确定性全图前向 × 2 位级一致断言、L2 归一、mappedID 行序断言、
           occupancy sidecar
INDEX  : build_index(既有代码,零改动)
         → HNSW M=16 efC=200 + exact-dot fidelity 硬门(r@10 ≥ 0.99)
```

### 1.2 冻结门的一次诚实降级(必须记录的工程事实)

原 2K phase0_freeze 的门是**位级 sha 一致**(重训 state_dict 逐位等于记录)。
D0 上该门**不可达**:CUDA scatter 聚合内核非确定(原子加顺序),54.8K 边下
位级分歧几乎必然(12K 边的 2K 图是侥幸可复现)。处置:

- 冻结门降级为**功能等价**:同 split 五指标 replay 与记录聚合的
  max |Δ| ≤ 0.02;
- 两个 sha(记录/重训)与 `state_dict_sha256_match=False` 均如实写入
  checkpoint 的 repro 字典——不隐藏分歧,只改判据。

**实测:replay max |Δ| = 0.0000**——权重位级不同、指标到小数点后四位完全一致,
功能门以满分通过(CUDA 非确定性未触及任何评测行为)。

### 1.3 结果(全绿)

```
checkpoint : stage2TrainGraphSAGE/artifacts/ablation/d0/ckpt/L1L3b_s0_i0.pt
             (family_vocab 371 行 + task_type_vocab 17 行双 sidecar 绑定)
export     : stage3HNSW/artifacts/exports/d0_L1L3b/
             z_m [9491×128] + z_d [1463×128],graph sha 45c1dd3c…,
             forward 位级确定性 ✅  行序断言 ✅  spotcheck 20/20 ✅
index      : stage3HNSW/artifacts/indexes/d0_L1L3b/(6.3 MB,0.52s 构建)
fidelity   : r@1 = r@10 = r@50 = r@100 = **1.000**(efS=200,全部 1,463 个 z_d 查询)
roundtrip  : 磁盘重载 k=50 检索结果与内存索引逐位一致 ✅
```

**Stage-3 的卖点兑现:** 从 2K/G2 换到 5 倍规模的 D0/L1L3b,
export/build/query 三段代码**一行未改**,fidelity 照旧 1.000。

## 2. W3 卫生行 —— D2-P2 温和 degree-cap

### 2.1 定义(v2 §6 P2 规格照抄,首次落地)

```
tau = P95(消息图 trained_on 模型出度 | 带标签模型)
超过 tau 的模型:按边权(accuracy)保留 top-tau 条消息边;其余模型不动
监督(edge_label_*)与 lineage 边永不触碰——只剪消息图
对 split 三视图(train/val/test 消息图)分别施加;确定性(稳定排序)
```

配置:L1L3b + `degree_cap_q=0.95`,seeds 0–2,对照 `w1_baselines` 同 seeds
的 L1L3b。**预期中性**(E9/E13:病灶在监督分布,不在消息聚合)——本行是
验证性卫生检查,不是涨点行。

### 2.2 代码改动

| 文件 | 改动 |
|---|---|
| `stage2TrainGraphSAGE/graph_surgery.py` | **新增** `degree_cap_trained_on(data, quantile)`(P2 温和版;REV 镜像;返回 stats) |
| `stage2TrainGraphSAGE/w1_dzero.py` | 行 `L1L3bP2`(`degree_cap_q=0.95`);切分后对三视图消息图施加(打印 cap stats) |
| `stage3HNSW/d0_freeze_export.py` | **新增**(§1.1 流程;功能性冻结门) |

## 3. 复现

```bash
cd ModelLakeFishing  # 仓库根的 codes/ 下
# Stage-3 全链(冻结 ~2 分钟 GPU + 导出/建索引 ~1 分钟)
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.d0_freeze_export
# W3 卫生行(3 次训练)
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero \
    --rows L1L3bP2 --seeds 0 1 2 --tag w3_hygiene
```

## 4. W3 结果(seeds 0–7,8-seed 终判:**中性,不采用**)

产物:`artifacts/ablation/d0/{w3_hygiene,w3_ext}/L1L3bP2.json`。
cap 实况:tau=27(P95),162 个模型被剪,但消息边 **42.5K→16.5K(−61%)**
——重尾分布下"温和版"在边量上并不温和(162 个 hub 持有六成消息边量)。

**过程的诚实记录(3-seed 假信号 → 8-seed 均值回归):**
seeds 0–2 曾给出 root gold@10 0.087→0.127(三 seed 全升、flat Δgold@10 CI
[+0.119, +0.203])的强信号,一度看似推翻"消息图非病灶"的判断;
补 seeds 3–7 后信号消失——**正是 E10 纪律(n 不足不裁决)防止了一次误晋升**。

**8-seed root_macro(正式口径):**

| 指标 | L1L3b | L1L3b+P2 | per-seed |
|---|---|---|---|
| hit@1 | **0.312±0.037** | 0.302±0.055 | — |
| top3 | **0.527±0.023** | 0.499±0.051 | — |
| regret@1 | **0.055** | 0.058 | — |
| gold@1 | **0.027** | 0.018 | — |
| gold@10 | **0.100±0.058** | 0.097±0.049 | P2 胜 4/8 |

flat 配对(8-seed):Δhit@1 [+0.043, +0.101] 与 Δgold@10 [+0.009, +0.057] 为正,
但 Δtop3 [−0.068, −0.028] 显著负、Δgold@1 偏负——flat 的"增益"又是大根族
构成效应(与 W1 §8 同一机理,方向相反地作用于 P2),按根公平计权后全部持平略负。

### 终判

- **P2 不采用;L1L3b 维持唯一晋升配置**(Stage-3 索引无需重建,§1.3 已定稿);
- v4 的"预期中性"在正式口径下**被 8-seed 确认**——消息图卫生在 D0 上
  不伤害也不增益,病灶在监督分布的判断(E9/E13)边界依然成立;
- 记录在案的开放事实:P2 的 flat hit@1 提升 [+0.043, +0.101] 真实存在于
  兄弟节点密集的根族内部——若未来 serving 侧对 per-language 族内精排有
  独立需求,P2 可作为该场景的专用旋钮重访(排 47K 之后)。

### v4 全链收官状态

```
v3 诊断(监督盲区)→ L1 破天花板(2K 0.046→0.128)→ Z 轨证伪(表示=目标的影子)
→ v4 收束 → W2 精修(L1L3b,hit@1 回收)→ W1 D0 裁决(root 口径晋升,权衡不存在)
→ Stage-3 重索引(fidelity 1.000,管线零改动)→ W3 卫生(中性确认,不采用)
```

v4 排期全部执行完毕。剩余远期项:47K(pyg-lib + z 缓存 + alibi 分位复核)、
gpt-neo 探针视图回补(仅当 z_d 条件化再成瓶颈)、F1b/Z 重访(门槛条款约束)。
