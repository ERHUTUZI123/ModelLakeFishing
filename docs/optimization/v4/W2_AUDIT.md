# v4 W2 可审计记录 —— lake 目标精修(L1b 分母加密 / L4 IPW 正样本 / L2b alibi 硬负样本)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 W2(v4)
**前提:** v4 判定唯一有效杠杆是监督分布(E13);L1L3(gold@10 0.128)为冻结候选。
W2 = 在这个杠杆上继续拧的三个原理性螺丝,全部对照 L1L3,one change per row。

---

## 1. 行定义与动机

| 行 | 改动(相对 L1L3) | 原理 |
|---|---|---|
| L1b | `global_n_neg` 64 → 256 | 采样 softmax 的分母估计方差随样本数 ~线性下降;最便宜的一行(纯 cfg,零代码) |
| **L4** | 正样本项 IPW 加权 w(m) = (deg(m)+1)^−0.5 | **v4 唯一新理论杠杆:** 负样本侧已被 logQ 校正,但**正样本侧的 hub 偏置原封未动**——top-frac 正样本集合里高度数模型占比高,每个正样本项都在把查询方向拉向自己。度数逆倾向加权是 logQ 的精确对偶,闭合该目标内最后一个已知偏置源 |
| L2b | 一次性 **alibi** 硬负样本(ep10,P90 度数 ∨ 任务不符) | L2 自我拆台的修复:只有"有流行度嫌疑或任务不符"的现胜者才可作硬负样本;低度数且任务相符的挖掘命中(**隐藏 gold 的精确特征**)永不入负样本集 |

## 2. 数学定义

### 2.1 L4 —— IPW 正样本(logQ 的对偶)

```
L(d) = − Σ_{p∈P_d} w(p)·log softmax_p / Σ_{p∈P_d} w(p),   w(m) = (deg(m)+1)^−β,β=0.5
```

与负样本侧的对偶关系:logQ 校正让 hub 作为**负样本**时被按度数折减
(`s − log q`,q∝(deg+1)^0.75);L4 让 hub 作为**正样本**时同样按度数折减。
β=0.5 弱于负侧 0.75:正样本是真监督,折减过猛会浪费标签
(2K 真图 w 范围 [0.24, 1.0],hub 正样本权重约为 tail 的 1/4)。
`pos_ipw=None` 时逐位等于 L1L3(测试断言 uniform IPW == 无 IPW)。

### 2.2 L2b —— alibi 筛选

```
alibi(m, d) = deg(m) ≥ P90(deg | 带标签模型)          流行度嫌疑
            ∨ (profile(m) ≠ ∅ ∧ task(d) 已知 ∧ task(d) ∉ profile(m))   任务不符
H_d = 按分数深挖 3×hard_k 后按 alibi 过滤,截前 hard_k;正样本永远排除
profile(m) = 训练可见 trained_on 边指向的已知任务集合(与 G-pools 同源)
```

对照 L2 原版:top-20 现胜者里**低度数、任务相符、无标签**的模型
(= 隐藏 gold 的精确特征)在 L2 会被压下去,在 L2b 被豁免。
2K 真图实测:带标签模型 1,377,P90 度数 = 11(max 16)——阈值有实义,
不是形同虚设。task(d) 未知(修复后仍 ~28%)时仅剩度数一条 alibi,保守方向正确。

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `stage2TrainGraphSAGE/losses.py` | `global_lake_loss` 新增 `pos_ipw` 参数(加权正样本平均,None=逐位旧行为);**新增** `mine_alibi_hard_negative_sets`(3× 深挖 + alibi 过滤)与 `build_model_task_profiles`(任务画像,G-pools 同源逻辑复用) |
| `stage2TrainGraphSAGE/train.py` | 挖掘分支:`global_ctx["alibi"]` 存在时走 alibi 挖掘(打印带 tag);lake 损失调用透传 `pos_ipw` |
| `stage2TrainGraphSAGE/ablation.py` | cfg 键 `pos_ipw_beta`(建 w 并打印范围)、`hard_alibi`/`alibi_deg_quantile`(组装 deg/任务画像/数据集任务 id 进 ctx) |
| `stage2TrainGraphSAGE/w2_phase.py` | **新增**:四行 runner(L1L3 对照),tag `v4_w2phase` |
| `stage2TrainGraphSAGE/tests/test_w2_phase.py` | **新增**:5 项机制测试 |

## 4. 机制验证(5/5 通过;L/Z 轨回归 12/12 不破)

1. `pos_ipw=None` 与 uniform 权重逐位一致(候选行为零漂移保证);
   非平凡权重确实改变损失;
2. **IPW 方向性**:构造 hub 正样本反向对齐、tail 正样本正对齐的场景,
   IPW 后损失下降——hub 的坏项按设计被折减;
3. **alibi 筛选逻辑**:高度数者保留、低度数任务不符者保留、
   **低度数任务相符者剔除(隐藏 gold 特征)**、无画像者剔除、正样本永不入集;
4. 数据集任务未知时仅按度数筛(不误用任务 alibi);
5. 任务画像构建正确(未知任务数据集不进画像)。

测试期间的一个如实记录:P90 阈值在**度数分布过平**时会退化
(合成测试首版 P90=1,deg=1 即算 hub)——真图 P90=11 无此问题,
但 D0 重尾分布下 `alibi_deg_quantile` 需随分布复核(已列入 v4 §0.6 风险 2)。

## 5. 复现

```bash
cd ModelLakeFishing/stage2TrainGraphSAGE
../.venv/Scripts/python.exe -m pytest tests/test_w2_phase.py -q    # 5 passed
cd ../..
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w2_phase
# 产物: stage2TrainGraphSAGE/artifacts/ablation/top1/v4_w2phase/{L1L3,L1b,L4,L2b}.json,
#        W2_report.{json,md}
```

## 6. W2-phase 结果(2026-07-19,12 次训练完成)

产物:`artifacts/ablation/top1/v4_w2phase/{L1L3,L1b,L4,L2b}.json, W2_report.{json,md}`。
形式判定三行全 REJECT——**这是 n=61 功效下的预期输出**(v4 §0.2-3:2K 只筛方向,
不作数值裁决);方向信息本身非常清晰:

| 行 | hit@1 | top3 | regret@1 | gold@10 | τ | z_d PR | Δgold@10 CI |
|---|---|---|---|---|---|---|---|
| L1L3(对照) | 0.338±0.033 | 0.642 | 0.057 | 0.128±0.058 | 0.278 | 3.31 | — |
| **L1b(n_neg 256)** | **0.401±0.042** | **0.679** | **0.048** | **0.155±0.029** | **0.303** | **3.40** | [−0.028, +0.092] |
| L4(IPW 正样本) | 0.331 | 0.616 | 0.061 | **0.066** | 0.281 | 3.31 | **[−0.110, −0.018] 显著负** |
| L2b(alibi 挖掘) | 0.338 | 0.651 | 0.058 | 0.118 | 0.253 | 3.16 | [−0.055, +0.037] |

### 读数

1. **L1b 是方向赢家,且可能瓦解了 E12 权衡轴。** n_neg 64→256 后:
   hit@1 **完全恢复到 0.401**(= G2 的 0.400,hit_ok 通过)、top3 0.679 与
   regret 0.048 均为**全项目历史最优**、gold@10 名义最高 0.155(std 还从
   0.058 收窄到 0.029)、τ 0.278→0.303 部分回升、z_d PR 保持 3.40。
   机理:更密的负样本使全局项的分母估计方差骤降,梯度噪声不再迫使模型
   在"组内细排"与"全局分离"之间二选一。**v3 以来的 hit@1 代价可能不是
   权衡的本质,而是 n_neg=64 的估计噪声**——这一假设由 D0 终裁。
2. **L4 证伪(CI 显著负)。** IPW 正样本把 gold@10 打回 0.066。事后读法:
   负样本侧的 logQ 校正的是**采样分布**(hub 被采得多所以折减),而正样本
   是**观测事实**——hub 模型进入 top-frac 是因为它真的在该数据集上表现好,
   按度数折减正样本等于按流行度折扣真监督。对偶性在形式上成立、在语义上
   不成立。v4 §0.2-1"正样本侧"杠杆就此关闭,负结果入 §0.4 字典。
3. **L2b 修复成功但增益为零。** alibi 确实消除了 L2 的自我拆台
   (0.083→0.118 ≈ 对照),但 logQ 全湖采样已经覆盖了硬负样本想提供的信号
   ——挖掘管线整体退役,不再进入任何后续行。
4. z_m PR 四行 3.16–3.40 恒定(第四次确认)。

### 结论

- **携带配置更新:L1L3 + n_neg 256(记 L1L3b)取代 L1L3 作为进入 W1/D0 的
  唯一携带配置**——它在五个 serving 指标上全部 ≥ L1L3,无任何代价面;
  正式晋升仍待 D0(gold@10 的 +0.027 在 2K 上不可裁决)。
- L4/L2b 关闭;监督分布杠杆在 2K 上已拧到头,剩余的一切问题
  (L1b 增益真伪、E12 权衡余量、规模成立性)= **W1 D0 的裁决范围**。
