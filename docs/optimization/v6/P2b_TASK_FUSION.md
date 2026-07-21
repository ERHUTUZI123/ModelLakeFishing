# v6 P2b 可审计记录 —— serving-time task-prior 融合(零重训,采纳)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 T 轨 P2b
**性质:** 纯 serving 层重排,零重训。与 S2 正交的第二个 serving 通道——
S2 用 sibling(同根)先验,P2b 用 task(同任务)先验。**决定 serving 层融合
一个还是两个通道。**

---

## 1. 机制与动机

P0 发现 predictable 层(9% 根)O-task@10 = 0.87 但现任仅 0.31——**task 元数据
未用足**。P2b 在 serving 侧补:除 MIPS 外,用模型在**同任务训练数据集**上的
收缩均值精度作先验:

```
task_boost(m) = (Σ_{d'∈同任务训练集, d'≠D} acc(m,d') + prior·k) / (n + k)   (k=5 收缩)
fused(D,m) = minmax(z_d·z_m)[m] + β · task_boost(m)
```

task(D) 是查询携带的**元数据**(task_type_id),非 held-out 标签;boost 只用
训练可见同任务数据集、排除 D 自身——合法(§0.6-3)。

## 2. 代码改动

| 文件 | 性质 |
|---|---|
| `stage2TrainGraphSAGE/p2b_task.py` | **新增**:复用 s2_sibling 的 node-level split;task-prior 收缩均值融合;β 扫描;同时算 MIPS / +task / +sibling(S2)/ +sibling+task 四种,按 has_sibling 分层 |

## 3. 8-seed 终裁(2026-07-19)—— **采纳,与 S2 叠加**

root_macro,8 seeds。tag:mips=纯 MIPS,t1.0=P2b(β=1),sib=S2(α=1),
sibtask=S2+P2b。

| 分层 | n/seed | mips | t1.0(P2b) | sib(S2) | **sibtask(S2+P2b)** |
|---|---|---|---|---|---|
| all | 123 | 0.137 | 0.208 | 0.188 | **0.219** |
| has_sibling | 93 | 0.158 | 0.239 | 0.259 | **0.261** |
| **no_sibling** | 30 | 0.121 | **0.180** | 0.121 | **0.180** |

**两个决定性判读(均 per-seed 一致 + 配对显著):**

1. **P2b 兑现了 S2 够不到的孤立层(v6 的关键补全)。** no_sibling 层
   MIPS 0.121 → P2b **0.180**(6/8 seeds,配对 Δ=+0.062,CI [+0.025, +0.100]
   排除 0);**S2 在该层 = 0.121 不变**(无兄弟,正确不触发)。**这正是 v5 内容
   通道证伪失败的那 18% 孤立根**——P2b 的 task 通道补上了。
2. **S2 与 P2b 叠加(不同于 S1 的替代)。** all 层 sibtask **0.219 > sib
   0.188**(7/8 seeds)。两者互补:sibling-rich 上 sibling 主导(sibtask 0.261
   ≈ sib 0.259,task 微增),孤立层上 task 独撑(sibtask = t1.0 = 0.180)。

### 为什么 P2b 叠加而 S1 不叠加(统一 v6 教训)

| | S1(不采纳) | S2 / P2b(采纳) |
|---|---|---|
| 侧 | 训练侧(全局池化) | serving 侧(外科式加法) |
| 副作用 | 模糊表示、税孤立层 | 不适用处 boost≡0、零副作用 |
| 叠加 | 与 S2 冗余有害 | S2(sibling)+P2b(task)互补叠加 |

**统一律:serving 侧、外科式、加法融合的合法湖标签先验可叠加;训练侧全局
池化不行。** 且 sibling(同根)与 task(同任务)是两个**覆盖互补**的通道——
sibling 管 82% sibling-rich,task 管含孤立层在内的全部(同任务几乎总存在:
no_task_prior 层为空)。

## 4. 采纳决定与 serving 策略

**采纳 P2b(β=1.0),与 S2(α=1.0)共同进 serving 栈。** 统一重排:

```
fused(D,m) = minmax(z_d·z_m) + α·sibling_boost(m) + β·task_boost(m)
             α=1(有兄弟处主导),β=1(孤立层独撑,兄弟处微增)
```

**这完成了 v6 的 E19 分叉的可部署闭环:**
- sibling-rich 根(82% gold):S2 主导(+ P2b 微增);
- 孤立根(18% gold):P2b 兑现——**v5 内容通道失败的部分,由 task 通道补上**。

serving 层无分支:两个 boost 各自在不适用处自动归零,一套逻辑覆盖两类根。

## 5. 复现

```bash
cd ModelLakeFishing  # 仓库根
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.p2b_task \
    --seeds 0 1 2 3 4 5 6 7 --epochs 25
# 产物: stage2TrainGraphSAGE/artifacts/ablation/d0/p2b_task/p2b_report.json
```