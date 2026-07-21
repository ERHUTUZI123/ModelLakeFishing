# v3 Z 轨可审计记录 —— z_d 条件化重建(Z1 探针降维 / Z2 数据集斥力)

**日期:** 2026-07-18
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3(v3 修订,Z 轨)
**姊妹篇:** `L_TRACK_AUDIT.md`(L 轨已完成:L1L3 gold@10 0.128 冻结为候选)
**动机一句话:** z_d 内在维度 ≈ 2、候选重叠 57.6%、静态榜 0.656@500——查询向量
表达不出"我是哪个任务的哪种数据"。模型侧特征改完没用(E7),条件化的另一半在查询侧。

---

## 1. 事实修正(比方案原文更极端)

方案 §0.3 写"gpt-neo 探针 768 维";实测图内 `data['dataset'].x` 为 **4618 维**
(gpt-neo 多视图 e_domain 3840 + e_label 384 + e_card 384 + e_stats 10)。
冻结 : 可学习 = 4618 : 28 = **99.4% : 0.6%**——容量霸权比方案预估严重 6 倍,
Z1 的结构性理由更强。

## 2. 行定义

| 行 | 内容 | 对照 |
|---|---|---|
| Z0 | G2 复刻 | — |
| Z1 | Z0 + 冻结视图可学习投影 4618→128 | Z0 |
| Z2 | Z0 + task 修复 + push-apart(修复是 Z2 声明依赖,方案原文即如此;归因注意事项见 §5) | Z0 |
| L1L3 | v3 候选复刻(栈行的第二参照) | — |
| Z1L | L1L3 + Z1 | L1L3 |
| ZL | L1L3 + Z1 + Z2(v3 全栈) | L1L3 |

## 3. 数学定义

### 3.1 Z1 —— 冻结视图的可学习投影(结构性再平衡)

```
x_d 拼接 = [ W_p · x_frozen ‖ E_task[t] ‖ E_nclass[c] ‖ E_arity[a] ]
W_p ∈ R^{128×4618}(nn.Linear 无 bias,可学习)
离散占比:28/4646 = 0.6%  →  28/156 = 18%
```

与 F4 的 name 投影同一结构性论证(concat+Linear 下标量降权被吸收、
维度不可被吸收),但**可学习**而非冻结随机:探针视图信息丰富但高度冗余,
让训练自己选出存活子空间;xd0 离散描述符(task/n_class/arity)从
被 4618 维淹没变为占 18% 的一等公民。

### 3.2 Z2 —— 数据集-数据集 push-apart(斥力项)

```
L_push = mean_{a,n: task(a)≠task(n), 两者已知}  relu(⟨z_a, z_n⟩ − margin)
margin = 0.2;每步采 32 anchor × ≤16 异任务负样本;task=Other(0)永不参与
```

**为什么是 hinge 不是 InfoNCE:** 斥力单向设计——我们断言"不同任务应可分",
**不**断言"同任务数据集可互换"(后者是塌缩方向)。**为什么需要它:** 现在
z_d 上没有任何互斥力:similar_to 边只拉近,trained_on 监督只经共享模型间接
塑形——内在维度 2 是单向力的直接后果。斥力项给 z_d 空间第一个展开的理由。
Other 豁免防错误斥力(修复后仍有 ~28% 未知任务节点)。

### 3.3 梯度边界与实现位置

Z2 作用在**训练消息图的全图 z_d** 上(与 L1 的 global 项共享同一次
full-graph forward,零额外前向开销);Z1 是 encoder 内部结构,
checkpoint 契约照旧(arch 记 `dataset_frozen_proj_dim`,旧 ckpt 默认 None)。

## 4. 代码改动清单

| 文件 | 改动 |
|---|---|
| `stage1BuildTransferGraph/dataset_embed/model_node_encoder.py` | `DatasetNodeEncoder` 新增 `frozen_proj_dim` kwarg(可学习 `nn.Linear` 无 bias;默认 None = legacy 逐位一致) |
| `stage2TrainGraphSAGE/model.py` | `HeteroGraphSAGE` 透传 `dataset_frozen_proj_dim` |
| `stage2TrainGraphSAGE/learnable.py` | arch 记录/重建 `dataset_frozen_proj_dim`(旧 checkpoint 兼容) |
| `stage2TrainGraphSAGE/losses.py` | **新增** `dataset_push_apart_loss`(异任务 hinge 斥力,Other 豁免,采样有界) |
| `stage2TrainGraphSAGE/train.py` | `zpush_ctx` 参数;full-graph z 在 global/zpush 间复用(`use_global or use_zpush` 才前向) |
| `stage2TrainGraphSAGE/ablation.py` | cfg 键 `dataset_frozen_proj_dim` / `lambda_zpush` / `zpush_margin` / `zpush_n_anchor` / `zpush_n_neg` |
| `stage2TrainGraphSAGE/z_phase.py` | **新增**:六行 runner,双参照门禁(Z1/Z2 vs Z0;Z1L/ZL vs L1L3),z_d PR 全行报告 |
| `stage2TrainGraphSAGE/tests/test_z_phase.py` | **新增**:5 项机制测试 |

## 5. 机制验证(5/5 通过;D1/L 回归 17/17 不破)

1. Z1 投影维度算术 + **在参数组内**(与 F4 冻结 buffer 相反,断言区分)+ 梯度流;
2. 斥力对筛选:只跨异任务对;全同向量 → 正损失,正交 → 0(margin 语义);
3. 全 Other → 零损失 no-op;梯度流向 z_d;
4. 同任务对永不互斥(塌缩方向不受推力);
5. 真图全模型 + checkpoint roundtrip(z_d 输出 allclose,proj 维度经 arch 重建)。

**归因注意事项(如实):** Z2 行捆绑 task 修复(其声明依赖)。已知 L3 单独在
pools 模式 REJECT(L 轨 §6.3),因此 Z2 vs Z0 的差值 = 修复 + 斥力的净效应;
若 Z2 优于 Z0 而 L3 劣于 L0(同底对照),斥力的正贡献下界即可推出。

## 6. 复现

```bash
cd ModelLakeFishing/stage2TrainGraphSAGE
../.venv/Scripts/python.exe -m pytest tests/test_z_phase.py -q     # 5 passed
cd ../..
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.z_phase
# 产物: stage2TrainGraphSAGE/artifacts/ablation/top1/v3_zphase/{Z0,Z1,Z2,L1L3,Z1L,ZL}.json,
#        Z_report.{json,md}
```

## 7. Z-phase 结果(2026-07-18,18 次训练完成)—— 全行 REJECT

产物:`artifacts/ablation/top1/v3_zphase/{Z0,Z1,Z2,L1L3,Z1L,ZL}.json, Z_report.{json,md}`。

| 行 | hit@1 | gold@10 | z_d PR | z_m PR | 判定(对照) |
|---|---|---|---|---|---|
| Z0(G2 对照) | 0.400±0.062 | 0.046±0.035 | 2.4 | 3.1 | 对照 |
| Z1(投影 128) | 0.301±0.054 | 0.062±0.052 | **1.8 ↓** | 2.8 | REJECT(Z0) |
| Z2(修复+斥力) | 0.311±0.020 | 0.064±0.025 | 2.5 | 2.9 | REJECT(Z0) |
| L1L3(候选) | 0.338±0.033 | **0.128±0.058** | 3.3 | 3.4 | 对照 |
| Z1L | 0.366±0.037 | 0.065±0.015 | 3.2 | 3.2 | **REJECT(L1L3)** |
| ZL(全栈) | 0.377±0.037 | 0.065±0.015 | 2.9 | 3.1 | **REJECT(L1L3)** |

关键 bootstrap:Z1L/ZL 对 L1L3 的 Δgold@10 CI = **[−0.119, −0.018] 显著为负**
——不是噪声,是叠加 Z 真的摧毁了 L 轨一半的增益。

### 读数(假设被证伪,如实记录)

1. **Z1 的"容量霸权"论证在 2K 上不成立,方向反了。** 投影 4618→128 后
   z_d PR 从 2.4 **降到 1.8**——压缩不仅没把离散描述符扶正,反而进一步
   压塌了 z_d(4618 维冗余视图里显然还承载着可分性,一刀切到 128 把它切掉了)。
   gold@10 名义 0.046→0.062 但 CI 跨 0,且 hit@1/regret 全回退。
2. **Z2 斥力项方向性存疑:z_d PR 仅 2.4→2.5。** hinge(margin 0.2)满足即
   梯度归零——异任务对被推到 cos<0.2 后就停手,展开有限;gold@10 CI 跨 0。
   作为"z_d 第一个互斥力"的机制探针它工作了(损失非零、有梯度、有 no-op 保护),
   但强度/形态不足以改变几何。
3. **最重要:Z 叠加在 L1L3 上是净伤害(−0.06 gold@10,CI 排除 0)。**
   读法:L1 的 logQ 全湖目标正是在**利用**高维 z_d/z_m 空间来分离 gold 与 hub;
   Z1 把 z_d 输入压到 128 维、Z2 再施加任务粒度的几何约束,恰好没收了
   这种自由度——hit@1 回到 0.377(hit_ok True)说明模型退回了
   "带标签候选排序"的舒适区。**目标函数(L)与表示容量(Z)在 2K 上是
   竞争关系,不是互补关系。**
4. z_m PR 全表 2.8–3.4 照旧不动(第三次确认)。

### 结论

- **Z 轨在 2K 图上证伪:Z1/Z2 以现参数不进入候选,L1L3 维持唯一 v3 候选。**
- 有据的重访条件(记录,不自动执行):(a) D0 图上任务面宽 5 倍、613 gold
  节点,z_d 条件化的可学习支点变多,Z 假设值得在 D0 基线后用一行复检;
  (b) Z1 放宽到 512 维或加残差旁路;(c) Z2 换有温度的 InfoNCE 全斥力
  (梯度不因 margin 满足而消失)并加大 λ。三者都排在 D0 之后。
- v3 主线不变:**下一步 = D0 建图 + L1L3 复跑**(方案 §0.5 第 3 步)。
