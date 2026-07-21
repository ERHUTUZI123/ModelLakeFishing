# v3 L 轨可审计记录 —— 训练目标重构(L1 logQ 全湖采样 / L2 硬负样本 / L3 任务元数据修复)

**日期:** 2026-07-18
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3(v3 修订,L 轨)
**动机一句话:** E8/E9 —— 28 个历史配置里唯一动过 clean gold@10 的杠杆是 loss 侧
(G1,0.010→0.054);而训练目标至今从未见过 serving 的失败对
(压过 gold 的 unlabeled hubs 从不作为负样本出现)。L 轨把这两个事实变成三行代码级干预。

---

## 1. 行定义(one change per row,G2 配置全冻结)

| 行 | 内容 | 关闭的缺口 |
|---|---|---|
| L0 | G2 复刻(任务不兼容池 + known-low,hard_frac 0.5) | 对照 |
| L3 | L0 + 数据集 task_type 修复 apply | 池覆盖 30/64 → **62/64**(E3 缺口) |
| L1 | L0 的全局项换成**全湖 logQ 校正采样 softmax**(弃池) | 监督盲区 E9:unlabeled hubs 首次成为负样本 |
| L1L3 | L1 + task_type 修复(修复同时作用于 z_d encoder 特征) | 两者叠加 |
| L2 | L1L3 + **一次性**硬负样本挖掘(ep10,top-20 现胜者) | serving 失败对直接入目标 |

## 2. 数学定义

### 2.1 L1 —— 全湖 logQ 校正采样 softmax

**提议分布(word2vec 温度化的度数分布):**

```
deg(m) = 训练可见 trained_on 标签度数
q(m) ∝ (deg(m) + n0)^α          α = 0.75, n0 = 1(零度模型仍可采样)
```

**损失(每数据集 d,P_d = 训练可见 top-frac 正样本,S_d ~ q 采 n_neg 个):**

```
L(d) = −(1/|P_d|) Σ_{p∈P_d} log  exp(s_p/T) /
        [ Σ_{p'∈P_d} exp(s_{p'}/T) + Σ_{n∈S_d\P_d} exp(s_n/T − log q(n)) ]
s = ⟨z_d, z_m⟩,T = 0.1,S_d 有放回采样后滤除 P_d
```

**为什么 −log q 是关键(Bengio & Senécal 采样 softmax 校正):** 负样本从 q
采样时,`exp(s/T − log q)` 的样本均值是全湖 softmax 分母的无偏估计——
**等价于每一步都对整个湖做 softmax**。hub 被采得多(q 大)但单次贡献被
log q 折减,净效应:每个数据集的损失都在把所有非正样本方向往下推,
**包括监督从未标注过的 hubs**——这正是 E9 盲区的原理性闭合,
也是 label-degree bias 的训练时对冲(双塔检索的标准解法)。

**对 G-phase"可靠负样本"原则的有据推翻:** G-phase 规定 unobserved ≠ 差,
负样本必须可靠(任务不兼容/已知低分)。v3 明确破例:InfoNCE 语义下负样本
断言的是"平均而言不如正样本相关",不是"已知差"——检索目标(排序)与
校准目标(打分)对负样本的要求不同。方案 §0.3 已记录此原则变更。

### 2.2 L2 —— 一次性 serving 硬负样本(MNS 风格混合)

```
第 10 epoch,无 dropout 的训练消息图上 no_grad 前向一次:
H_d = top-20 按 ⟨z_d, z_m⟩ 排序的模型 \ P_d      (= "当前压过标注最优的模型")
之后每步:从 H_d 采 ≤16 个入分母,logit 不做 logQ 校正
        (混合负采样 MNS 惯例:定向集合非 q 的抽样,严格无偏性换失败对直达)
只挖一次,永不重挖 —— P3 同款防自我强化循环纪律。
```

### 2.3 L3 —— task_type 修复 apply(既有 dry-run 的执行步)

`task_type_enrichment.py`(review-first,早已跑完且 gate PASS)产出的 patch:
310 个 Other 节点中 **210 个**有充分证据(三层证据制:model-index task >
dataset card task_id/identity > task_category;40 冲突 + 62 证据不足者保守留
Other)。apply = 运行时覆写 `task_type_id` + vocab 从 10 行扩到 **26 行**
(原 10 行 id 逐一保留,新任务 id 10–25),**图 .pt 零改动**;
DatasetNodeEncoder 的 task 表随 vocab 扩行(只影响本次新训练,旧 ckpt 不受扰)。
安全闩:被覆写节点必须当前为 Other,否则视为 patch/图错配大声失败;
重复 apply 会被幂等闩拒绝。

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `stage2TrainGraphSAGE/losses.py` | **新增** `build_lake_logq`(度数提议分布)、`global_lake_loss`(logQ 校正采样 softmax + MNS 硬负样本混入)、`mine_hard_negative_sets`(一次性 top-k 挖掘) |
| `stage2TrainGraphSAGE/train.py` | global 步双模式(`lake` in global_ctx 走 L1 损失);epoch==hard_mine_epoch 时无 dropout no_grad 挖掘一次(带打印) |
| `stage2TrainGraphSAGE/d1_features.py` | **新增** `apply_dataset_task_repair`(patch/vocab 运行时附加;旧 vocab 行保留断言;Other 前置断言;幂等闩) |
| `stage2TrainGraphSAGE/ablation.py` | cfg 键 `global_mode="lake"` / `lake_alpha` / `lake_n0` / `hard_mine_epoch` / `hard_k` / `n_hard` 组装 global_ctx(缺省 = pools 旧路径) |
| `stage2TrainGraphSAGE/top1_baselines.py` | cfg 键 `repair_dataset_task`:split 之前 apply,`xd0_cfg` 传入训练(encoder 表随扩) |
| `stage2TrainGraphSAGE/l_phase.py` | **新增**:L0/L3/L1/L1L3/L2 行 + G-phase 晋升门 + z_m PR 门,tag `v3_lphase` |
| `stage2TrainGraphSAGE/tests/test_l_phase.py` | **新增**:7 项机制测试(下节) |

## 4. 机制验证(7/7 通过)

1. `build_lake_logq`:q 归一、度数单调、零度可采样、α 温度化比值精确
   ((deg+1) 比 11:1 → q 比 11^0.75);
2. logQ 校正项确实进入负样本 logit,梯度流向 z_m 与 z_d;
3. 正样本永不做负样本(极端 q 下统计验证:采样后滤除);
4. `mine_hard_negative_sets`:恰为非正样本中 top-k,正样本排除;
5. 硬负样本集在损失内二次滤正样本(混入计数验证);
6. `apply_dataset_task_repair` 真图:**210 个节点 Other→具体任务**,vocab
   10→26 行且旧行逐一保留,重复 apply 拒绝;
7. **L3 修复把任务不兼容池覆盖从 30 → 62 个数据集**(全图边集口径;
   dry-run 预估 50 为 per-split 保守值)。

## 5. 复现

```bash
cd ModelLakeFishing/stage2TrainGraphSAGE
../.venv/Scripts/python.exe -m pytest tests/test_l_phase.py -q     # 7 passed
cd ../..
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.l_phase \
    --rows L0 L3 L1 L1L3 L2
# 产物: stage2TrainGraphSAGE/artifacts/ablation/top1/v3_lphase/{L0..L2}.json,
#        L_report.{json,md}
```

## 6. L-phase 结果(2026-07-18,15 次训练完成)

产物:`artifacts/ablation/top1/v3_lphase/{L0,L3,L1,L1L3,L2}.json, L_report.{json,md}`。

| 行 | hit@1 | top3 hit@1 | regret@1 | gold@1 | **gold@10** | z_m PR | 判定 |
|---|---|---|---|---|---|---|---|
| L0(G2 对照) | **0.400±0.062** | 0.652±0.025 | 0.052±0.008 | 0.000 | 0.046±0.035 | 3.1 | 对照 |
| L3(仅修复) | 0.311±0.060 | 0.606±0.037 | 0.057±0.010 | 0.010 | 0.029±0.040 | 2.7 | REJECT |
| L1(logQ 全湖) | 0.354±0.096 | **0.677±0.037** | **0.047±0.004** | **0.027** | **0.110±0.047** | 3.2 | INVESTIGATE |
| **L1L3** | 0.338±0.033 | 0.642±0.012 | 0.057±0.002 | 0.010 | **0.128±0.058** | 3.4 | INVESTIGATE |
| L2(+挖掘) | 0.354±0.077 | 0.650±0.055 | 0.051±0.005 | 0.018 | 0.083±0.024 | 3.3 | INVESTIGATE |

**paired bootstrap(对 L0,n=61×3 splits pooled):**

| 行 | Δgold@10 CI | Δhit@1 CI | Δtop3 CI |
|---|---|---|---|
| L1 | [0.000, +0.128](边界显著) | [−0.128, +0.037] 跨 0 | [−0.055, +0.110] 跨 0 |
| **L1L3** | **[+0.018, +0.156] 排除 0** | [−0.147, +0.027] 跨 0 | [−0.092, +0.073] 跨 0 |
| L2 | [−0.028, +0.101] 跨 0 | [−0.128, +0.037] 跨 0 | — |

### 读数

1. **历史天花板首次被击穿。** 28 个历史配置的 gold@10 从未超过 0.054;
   L1 一举到 **0.110(2.4×)**、L1L3 **0.128(2.8×,CI 排除 0,三 split 全升,
   z_m PR 不降)**;gold@1 从恒 0 变为非零(L1 0.027)。方案 §0.4 的方向验证
   标准第一条(越过 0.054)**达成**——E9 监督盲区诊断被实验证实:
   把 unlabeled hubs 变成负样本正是缺失的那块杠杆。
2. **hit@1 名义回退 0.046–0.062,但统计上不可分辨**(CI 全部跨 0;且 L1 的
   top3 hit@1 0.677 与 regret 0.047 均为全表最优)。这不是"付出了确定代价",
   而是 E10 的功效问题再次现身——**是否真的有 hit@1 代价,只能在 D0
   (613 gold 节点)上回答**。形式判定 INVESTIGATE 是门禁按字面执行的正确输出。
3. **L3 单独 REJECT,但 L1+L3 组合最优。** 修复过的 task 元数据在 pools 模式下
   反而全面回退(26 行新 task 表 + 池变化的净效应为负);在 lake 模式下它
   通过 z_d encoder 特征侧贡献了 0.110→0.128。**元数据修复的价值实现依赖
   目标函数形态**——单独一行的"one change"归因在这里出现了交互效应,如实记录。
4. **L2 挖掘反而低于 L1L3(0.083 vs 0.128)——设计缺陷已识别:**
   挖掘的 top-20"现胜者"只排除了带标签正样本,而**真正的 gold(未标注的
   最优特化模型)如果已经被 L1 推进了 top-20,会被 L2 当作硬负样本往下压**
   ——自我拆台。L2 需要重设计(如按静态先验/流行度过滤挖掘集,只保留
   高度数 hub)或放弃;当前证据不支持采用。
5. **对 D2 的间接确认:** L1 是纯 loss 改动,消息图未动,hub 几何(z_m PR)
   不变的情况下 gold@10 翻倍——进一步支持 v3 的判断:病灶在监督分布,
   不在消息传递。

### 结论与下一步(按方案 §0.5)

- **L1L3 冻结为 v3 候选配置**(`l_phase.l_configs()["L1L3"]`),
  等待 D0 图上的复跑给最终晋升/否决;
- L2 按第 4 条重设计后再入队;F1b/F3b 特征重访继续押后;
- 下一步 = §0.5 第 3 步:D0 建图 + 双契约 + pyg-lib + B0/G2/L1L3 三列基线
  (按根分层报告)。
