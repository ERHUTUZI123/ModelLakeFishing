# v4 W1 可审计记录 —— D0 落地:湖定稿、建图、root-aware 切分、三列基线

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 W1(v4 critical path)
**携带配置:** L1L3b(= logQ 全湖采样 softmax + n_neg 256 + task 元数据,W2 方向胜者)
**一句话:** 2K 上悬而未决的一切(L1L3b 增益真伪、hit@1 权衡、规模成立性)在这里裁决。

---

## 1. C1 湖定稿(执行的决策)

| 决策 | 取值 | 依据 |
|---|---|---|
| 模型口径 | **strict = has_model_index ∨ has_lineage → 9,491** | 用户定调 family-only 可弃;`d0_model_intake.csv` 一列 flag 切换 |
| 数据集口径 | 有 ≥1 条 bounded chosen-metric 观测(对 strict 模型)的全部节点 → **1,463**(gold 613 全入) | v4 §0.3 W1-C1 |
| 监督边 | `d0_observations.parquet` 直出,chosen-metric 单指标,**54,795** 条 | 42% 重复边教训在 parquet 层已闭合,建图零再修 |

## 2. 建图结果(`d0_build_graph.py` → `hgraph_d0_v1.pt`)

```
sha256: 45c1dd3cffa0a94bc6a5e34263b025c6b113033ced299e2367f2555d60e2c781
        (1-D edge_attr 修复重建后的定稿 sha;首版 [E,1] 形态的 1c6e3cf0… 已废弃)
models 9,491 | dataset nodes 1,463 | roots 421 | trained_on 54,795
lineage 990(2K 图仅 96)| similar_to 29,260(KNN k=20 over e_card)
gold nodes in graph: 613/613 | task vocab 17,Other share 3.1%
size unknown share 52.5% | families 371
```

**五边型契约与 2K 逐项一致**(trained_on/rev + similar_to + is_base_of/rev,
1-D edge_attr——首版误建 [E,1] 被 `topk_similar_to` 当场拒绝,契约起效)。

**关键质变:task Other share 362 图上 86% → D0 上 3.1%。** vocab 从观测的
dominant_task 原生构建(17 类,min_count 5)——**L3 的"修复"在 D0 上不再是
patch,是建图时就做对**;task 不兼容负样本池与 z_d task 表第一次有了全覆盖
的元数据地基。

**D0-v1 特征口径(诚实记录的偏离):**
- xm0 frozen 448 = e_name 64(MD5 hash,seed 42)‖ e_desc 384(MiniLM over
  元数据描述串:pipeline_tag+tags+datasets 清单——**不是** model card 正文);
- xd0 frozen 458 = e_name 64(seed 43)‖ e_card 384(MiniLM over 数据集描述串)
  ‖ e_stats 10(log 边数/runs、值分布、根规模、gold 位);
- **gpt-neo 探针视图(2K 的 3840 维)未重建**——需要逐数据集下载样本 +
  GPU 天级;E13(表示不是杠杆)+ Z1 证伪(压缩反伤)支持先行推迟;
  若 D0 上 z_d 条件化成为新瓶颈,此项为首个回补候选。
- n_class/arity 全 unknown(0)——v1 范围裁剪,元数据解析推迟。

## 3. D2 root-aware 切分(`d0_splits.py`,新实现)

**切分单位是根,不是节点**(v4 §0.2-4):tatoeba 112 个 per-language 节点、
massive 102 个兄弟节点若跨切分即近重复泄漏。实现:

```
根按 seed 洗牌 → 贪心装填 test(≥20% 边)→ val(≥10% 边)→ 其余 train
train 消息图 = train 边的 70%(30% disjoint 仅作监督);val 消息 = 全部 train 边;
test 消息 = train+val 边;负采样 1:1;held-out 边及其反向边不进任何消息图(内置断言)
```

实测(seed 0):监督正边 10,263/8,295/12,290,根 182/40/98 三侧互斥 ✅。
**代价如 v4 风险 4 预告:测试面按根粒度波动**(三个 seed 的可评测试数据集
127/148/213、根 47/26/94)——这是把泄漏换成诚实,按根宏平均报告吸收方差。

## 4. E1 训练侧规模化(实测后的决策)

- **pyg-lib 未装,LightLinkLoader 全邻居在 11K 节点图上实测可行**:
  batch_size 升到 1,024(54.8K 监督边 → ~38 批/epoch,与 2K 节奏一致),
  1 epoch × 3 split 全链路(含建 lookup/评测)55 秒——25 epochs 三行约 1 小时,
  z 缓存方案(v4 风险 1)在 11K 尺度**尚不需要**,47K 时再启用;
- lake logQ 在 D0 重尾上的实况(v4 风险 2 的实测回答):labeled 2,757、
  **max_deg 638**(2K 仅 13)、q_head 0.0061——α=0.75 温度化把最热 hub 的
  采样概率压在 0.6%,重尾未失控;
- cfg 增加 `batch_size` 透传(`ablation.py` 一行)。

## 5. 代码改动清单

| 文件 | 性质 |
|---|---|
| `stage1BuildTransferGraph/d0_build_graph.py` | **新增**:D0 建图器(C1 口径、观测直出监督边、双特征、五边型、双契约断言、report JSON) |
| `stage2TrainGraphSAGE/d0_splits.py` | **新增**:root-aware 固定切分(同根整根同侧;泄漏断言内置) |
| `stage2TrainGraphSAGE/w1_dzero.py` | **新增**:D0 三列基线 runner(B0/G2/L1L3b × 3 seeds;flat + **按根宏平均**双视图;L1L3b vs G2 晋升门) |
| `stage2TrainGraphSAGE/ablation.py` | `batch_size` cfg 透传(一行) |
| `stage1BuildTransferGraph/artifacts/d0_lake/d0_graph_report.json` | 产物:建图数字机读版 |

L1L3b 在 D0 上**无需 repair patch**(task ids 原生),配置 = `l_configs()["L1L3"]`
去掉 repair flag + `global_n_neg=256`——与 2K 携带配置的唯一差异是元数据来源,
语义等价。

## 6. 复现

```bash
cd ModelLakeFishing/stage1BuildTransferGraph
../.venv/Scripts/python.exe d0_build_graph.py       # ~5 分钟,零网络(MiniLM 本地)
cd ../..
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero
# 产物: stage2TrainGraphSAGE/artifacts/ablation/d0/w1_baselines/{B0,G2,L1L3b}.json,
#        D0_report.{json,md}
```

建图数字复核:

```python
import json
r = json.load(open(r'ModelLakeFishing\stage1BuildTransferGraph\artifacts\d0_lake\d0_graph_report.json'))
assert r['models'] == 9491 and r['dataset_nodes'] == 1463 and r['roots'] == 421
assert r['trained_on_edges'] == 54795 and r['gold_nodes_in_graph'] == 613
assert r['task_vocab'] == 17 and r['task_other_share'] < 0.04
```

## 7. 三列基线结果(E2,seeds 0–2 完成;8-seed 扩展进行中)

产物:`artifacts/ablation/d0/w1_baselines/{B0,G2,L1L3b}.json, D0_report.{json,md}`。
gold@K 分母 = **9,491 全湖**(绝对值不可与 2K 比较)。

| 行 | 视图 | hit@1 | top3 | regret@1 | gold@1 | gold@10 |
|---|---|---|---|---|---|---|
| B0 | root_macro | 0.180 | 0.428 | 0.095 | 0.000 | **0.000** |
| G2 | flat | **0.486±0.084** | 0.340 | 0.054 | 0.003 | 0.051 |
| G2 | root_macro | 0.353 | 0.480 | 0.071 | 0.007 | 0.054±0.049 |
| L1L3b | flat | 0.294±0.020 | 0.410 | 0.062 | 0.023 | 0.062±0.040 |
| L1L3b | root_macro | 0.335 | **0.522** | 0.067 | **0.042** | **0.087±0.070** |

**训练诊断:** z_d PR:B0 1.74 → G2 2.85 → **L1L3b 3.58**(E11 在 5 倍规模上
复现:目标塑形条件化)。3-seed 门禁(flat pooled):**REJECT**
(Δgold@10 CI [−0.035, +0.021] 跨 0;flat hit_ok/regret_ok False)。

### 3-seed 读数(分裂判决,如实)

1. **B0 地板确认:** 全湖 gold@10 = 0.000——不带全局项的目标在 9,491 湖里
   完全找不到 gold,基线三列的第一列有了定量意义。
2. **胜者随测试根构成翻转(核心不确定性):** per-split root-macro gold@10:
   s0(47 根)L1L3b **0.186** vs 0.045、s1(26 根)0.042 vs 0.000——L1L3b 大胜;
   s2(94 根)0.032 vs **0.118**——G2 大胜。两种目标各有擅长的根族,
   3 个 root-aware split 的功效不足以裁决,**已补种子 3–7(8-seed 扩展)**。
3. **hit@1 权衡的 D0 答案是"视图依赖":** flat 视图 G2 大胜(0.486 vs 0.294)
   ——但 root_macro 视图**打平**(0.353 vs 0.335,三 split 全如此)。
   即 G2 的 flat 优势集中在兄弟节点众多的大根族(它们主导 flat 平均),
   按根公平计权后优势消失。**v4 的按根口径在此不是装饰,是结论的分水岭。**
4. L1L3b 的 root gold@1 = 0.042(G2 的 6 倍)且 top3 全表最优——
   gold 轴的方向优势在正式口径下保持,但 ±0.070 的 split 方差要求更多种子。

## 8. 8-seed 终裁(seeds 0–7,G2 vs L1L3b 各 8 次训练)

产物:`w1_ext/{G2,L1L3b}.json` + 合并分析 `w1_baselines/D0_report_8seed.json`
(1,292 个 pooled 数据集评测,paired bootstrap)。

**flat 视图(逐数据集配对 CI):**

| Δ(L1L3b − G2) | CI | 判读 |
|---|---|---|
| gold@10 | **[+0.028, +0.063] 排除 0** | 显著优 |
| gold@1 | **[+0.009, +0.026] 排除 0** | 显著优 |
| top3 hit@1 | **[+0.022, +0.064] 排除 0** | 显著优 |
| regret@1 | [−0.010, +0.002] 跨 0 | 无代价 |
| observed hit@1 | **[−0.098, −0.033] 显著负** | flat 视图的真实代价 |

**root_macro 视图(正式口径,8 seeds):**

| 指标 | G2 | L1L3b | per-seed 配对 |
|---|---|---|---|
| hit@1 | 0.299±0.059 | **0.312±0.037** | +0.013(4/8,打平) |
| top3 | 0.472±0.036 | **0.527±0.023** | +0.055(6/8) |
| regret@1 | 0.062 | **0.055** | −0.007(优) |
| gold@1 | 0.009 | **0.027(3×)** | +0.018(5/8) |
| **gold@10** | 0.053±0.037 | **0.100±0.058(+89%)** | **+0.047(6/8)** |

per-seed root gold@10:L1L3b 胜 6/8(s2、s5 归 G2)——3-seed 时的"翻转"
在 8 seeds 下收敛为清晰多数。

### 终裁判定

**L1L3b 依 v4 正式口径(root_macro)晋升为 D0 默认候选:**
五指标全部 ≥ G2(hit@1 +0.013 不降、regret 反优),gold 轴在数据集配对层
显著(三项 CI 排除 0)。**flat hit@1 的 −0.085 是已定量的真实构成效应**:
G2 的组内排序优势集中在兄弟节点众多的大根族——它们在 flat 平均里权重超编,
在按根口径下不改变判定。此差异记录在案,serving 侧含义(per-language 亲缘
查询族内的精排)留给 λ/混合调优或 rerank 层,不阻塞晋升。

E12 权衡轴的最终答案:**在按根公平计权下,权衡不存在**(hit@1 打平、
regret 反优);它只在 flat 视图作为构成效应存在。

### G-D0 验收对照(方案 §4.3 → v4 W1)

| 项 | 状态 |
|---|---|
| 新图过 Stage-1 sanity + Stage-2 契约 | ✅(建图断言 + 1-D attr 契约当场纠错) |
| 可评 gold 数据集 ≥200 | ✅(gold 613 节点全入图;8 seeds 覆盖测试根 26–94/split) |
| B0 与 G2 基线重跑 + 三列报告 | ✅(B0 地板 gold=0;G2/L1L3b 双视图 + 8-seed 配对) |
| Stage-3 重导出/重索引 | ⏳ 待晋升配置最终冻结后执行(管线零改动) |
