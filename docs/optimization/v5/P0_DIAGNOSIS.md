# v5 P0 可审计记录 —— 诊断漏斗:oracle 分解、近平局结构、可达性分层

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 P0(v5)
**性质:** 零训练;8 个 root-aware seeds、1,292 个测试数据集评测的确定性重放。
**产物:** `stage2TrainGraphSAGE/artifacts/ablation/d0/p0_funnel/`
(`p0_report.json` 机读 / `p0_rows.csv` 逐行 / `p1_coverage_probe.json`)。

---

## 1. Oracle 分解(root_macro,8 seeds,gold 口径与 five_metric_eval 逐位一致)

| 排序器 | gold@1 | gold@10 | gold@50 | 说明 |
|---|---|---|---|---|
| O-static(流行度/收缩均值) | 0.019 | 0.048 | 0.093 | 训练可见边的全局收缩均值(k=5),无标签模型 −inf |
| O-task(同任务画像) | 0.047 | **0.080** | 0.253 | 同任务收缩均值;task 未知回退 static |
| **L1L3b(现任,记录值)** | — | **0.100** | — | 参照 |
| **O-sibling(故意泄漏上界)** | 0.059 | **0.563** | 0.772 | 测试根其余数据集的标签,leave-one-out |

## 2. 三个判决性结论

### 2.1 分叉裁决:信息饥饿证实(v5 §0.3-P0 的关键分叉)

**O-task(0.080)< L1L3b(0.100)** ——显式任务画像启发式打不过现任模型,
即 GNN 已经榨取了比"任务匹配 + 同任务均值"更多的现有信息。
"现有信息未被榨干、先走目标精修"的分支**关闭**;
**主攻确认为 P1(内容特征)+ P3(扩根)**。

**而天花板在 5.6× 之外:** O-sibling@10 = 0.563——只要知道测试根自己的标签,
gold@10 就是 0.563。0.100 → 0.563 的差距全部是"关于这个根的信息"的价值,
这正是 P1(数据长什么样)与 P3(更多根让跨根规律可学)要供给的东西。
**优化算法层面的余量已经很小,信息供给层面的余量是 4.6 倍。**

### 2.2 结构性上限:cold-gold 率 27.9%

**27.9% 的 gold 模型在训练可见图上零标签边**——其 z_m 只有 name/desc 特征与
lineage 消息,任何基于监督边的方法都结构性够不到。这单独把边法的 gold@10
封顶在 ~0.72,也是模型侧内容/特征信息(而非更多目标工程)的独立论据。
(另:20.2% 的测试数据集是单节点根,O-sibling 无定义,已在聚合中排除。)

### 2.3 E16 全面证实:严格 gold 是个大幅低估效用的噪声目标

近平局结构(1,292 数据集):

```
gold 与第 2 名的标签差:中位数 0.0017,P25 = 0(精确并列!)
70.3% 的数据集第 2 名在 gold−0.01 之内;79.0% 在 gold−0.02 之内
δ=0.01 的"够好集合"平均大小:7.5 个候选
```

用 s0 冻结 checkpoint 实测现任模型的 **gold-gap@10**(top-10 含 δ 内候选即中):

| 指标(s0) | flat | root_macro |
|---|---|---|
| gold@10(严格) | 0.118 | 0.186 |
| **gold-gap@10(δ=0.01)** | **0.551** | **0.351** |
| gold-gap@10(δ=0.02) | 0.567 | 0.354 |

**在产品语义下,系统已为 35% 的根(55% 的数据集)在 top-10 里放进了
"与最优差 ≤1%"的模型**——严格 gold@10 因为把近平局记 miss,
低估了约 3 倍。P4 的 δ 就此定为 **0.01**(70% 分位有据)。
注意 O-sibling 自身也只有 0.563 而非 1.0——完全知道根标签也钉不住
"唯一 gold"(per-language 波动 + 近平局),再次支持 gap 口径。

## 3. 可达性分层(附赠发现:一个便宜的混合机会)

按 O-task 根均值 ≥0.5 分层(456 个 root-seed 对):

| 层 | 占比 | O-task@10 | L1L3b@10 |
|---|---|---|---|
| predictable | 41(9%) | **0.870** | 0.311 |
| unpredictable | 415(91%) | 0.008 | **0.074** |

两个方向的信息互补:91% 的根上 GNN 优于任务启发式(用到了更多信号),
但**在任务画像本身就足以定位 gold 的 9% 根上,GNN 只拿到了 0.31/0.87**
——现任模型没有充分利用任务画像最强的那部分。这提示一行便宜的
**P2b 分数混融**(serving 侧 z-score 与 O-task 分数线性混合,零训练),
在 P1/P3 主线之外可平行验证。

## 4. P1 前提盘点:datasets-server 覆盖 52%

100 节点随机探针:`/splits` 端点 52/100 可用。P1 可直接采样约一半数据集,
其余回退元数据描述串——内容特征将是**部分覆盖**的增强视图,验收时按
"有样本/无样本"分层看增量。

## 5. 代码与复现

| 文件 | 性质 |
|---|---|
| `stage2TrainGraphSAGE/p0_oracles.py` | **新增**:三 oracle + cold-gold + 近平局 + 分层,全向量化,~2 分钟 |
| `artifacts/ablation/d0/p0_funnel/*` | 产物(报告 JSON/逐行 CSV/覆盖探针) |

```bash
cd ModelLakeFishing  # 仓库根 codes/
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.p0_oracles
```

复核片段:

```python
import json
r = json.load(open(r'ModelLakeFishing\stage2TrainGraphSAGE\artifacts\ablation\d0\p0_funnel\p0_report.json'))
o = r['oracles_root_macro']
assert round(o['task@10'][0], 3) == 0.080 and round(o['sibling@10'][0], 3) == 0.563
assert round(o['static@10'][0], 3) == 0.048
assert round(r['cold_gold_rate'], 3) == 0.279
assert round(r['gap_to_2nd']['share_le_0.01'], 3) == 0.703
```

## 6. 对 v5 排期的裁决(回填主方案 §0.5)

1. **分叉走 P1/P3 主线**(O-task < 现任,信息饥饿证实);P2(top-heavy
   精修)降为条件行;
2. **P4 落地参数:δ = 0.01**,gold-gap@10 即日起进所有报表作共同主指标;
   现任基线:root gold-gap@10 ≈ 0.35(s0 实测,8-seed 全量待 ckpt 补存);
3. 新增 **P2b 分数混融**(serving 侧,零训练)入队,针对 predictable 层的
   0.31 vs 0.87 缺口;
4. v5 数值承诺按纪律给出:**目标 = 关闭 L1L3b(0.100)与 O-sibling(0.563)
   差距的 25–40%**(即 root gold@10 0.22–0.29,等价 root gold-gap@10 ~0.45+),
   由 P1×P3 组合行在 8 seeds 上验收。
