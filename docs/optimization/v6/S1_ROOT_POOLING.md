# v6 S1 可审计记录 —— 根级正样本池化(训练侧,S2 的表示侧对偶)

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 S 轨 S1
**性质:** 训练侧改动——让 sibling 信号进入**表示**(z_d),而非只在 serving 融合。
S2(serving 层)的对偶:S2 在推理时借兄弟标签,S1 在训练时把兄弟偏好烧进 z_d。

---

## 1. 机制

lake 损失的全局项对每个数据集 d 用正样本集 P_d = 训练可见 top-frac 高分模型
(`M[:, d]`)。S1 把 P_d **按根池化**:

```
M_pooled[:, d] = OR_{d' ∈ root(d), 训练可见} M[:, d']
```

即"在该根**任一**兄弟数据集上是 top 的模型"都算 d 的正样本。这教 z_d 学
**根不变的模型偏好**——同根兄弟数据集应偏好同一批模型。

**合法性(root-aware 下天然安全):** M 由训练可见边构建,测试根的列全零、
池化后仍全零(其兄弟也在测试侧)。**只有训练根的正样本集变大,测试根的
held-out 标签绝不进入。** node-level 协议下(S2 场景),测试数据集 D 的列
由其**训练可见兄弟**填充(D 自己的边仍 held-out)——这正是 S1 的直接机制:
用兄弟标签塑形 z_d[D],而不碰 D 自己的标签。

## 2. 代码改动

| 文件 | 性质 |
|---|---|
| `stage2TrainGraphSAGE/losses.py` | **新增** `pool_membership_by_root(M, root_ids)`(根内并集广播,单例根不变,单调不缩) |
| `stage2TrainGraphSAGE/ablation.py` | `train_eval_one` 在 M 构建后按 `cfg['root_pool_positives']` + `cfg['root_ids']` 池化 |
| `stage2TrainGraphSAGE/w1_dzero.py` | 行 `L1L3bS1`;main 注入整数 root code |
| `stage2TrainGraphSAGE/s2_sibling.py` | `--s1` 标志:node-level 协议下训练 S1,评测同时给 S1-alone(α=0)与 S1×S2(α>0) |

机制测试通过:根内并集广播正确、单例根不变、`M_pooled ≥ M`(单调)。

## 3. 两个评测口径

- **node-level(via `s2_sibling --s1`,主):** S1 直接用训练兄弟塑形测试 D 的 z_d;
  与 S2 融合叠加。对照 = 无 S1 的 s2_sibling 基线。
  - S1-alone = (S1, α=0) vs (base, α=0):表示侧单独效应;
  - S1×S2 = (S1, α=1) vs (base, α=1):是否与 serving 融合叠加增益。
- **root-aware(via `w1_dzero L1L3bS1`):** 标准 D0 冷根协议,S1 只塑形训练根
  (对冷测试根是间接的根不变归纳偏置)。对照 = L1L3b。

## 4. 8-seed 终裁(2026-07-19)—— **S1 不采纳;S2 是 sibling 通道的正确交付方式**

**node-level(root_macro,8 seeds):**

| 对比 | base | S1 | per-seed |
|---|---|---|---|
| has_sibling α=0(S1-alone vs 纯 MIPS) | 0.162 | 0.195 | S1 上 **4/8**(掷硬币) |
| has_sibling α=1(S1×S2 vs S2) | **0.266** | 0.232 | S1 上 **2/8**(S1 拖累) |
| no_sibling α=0(孤立层) | 0.116 | **0.065** | S1 上 **2/8**(S1 伤害) |

**root-aware(vs L1L3b,8-seed root_macro):** gold@10 0.100→**0.095**、
gold@1 0.027→**0.007**、top3 0.527→0.500、gap@10 0.148→0.168、hit@1 0.312→0.320
——中性偏负。

### 裁决:REJECT。三条证据链

1. **S1-alone 是噪声**(has_sibling 0.162→0.195 但仅 **4/8** seeds)——
   证明兄弟偏好**能**烧进 z_d,但幅度不可靠(对比 S2 的 8/8);
2. **S1 与 S2 是替代品而非互补**:S1×S2(0.232)**低于**单独 S2(0.266),
   6/8 seeds S1 拖累。二者注入同一 sibling 信号——S1 在表示层把同根成员拉向
   共享偏好、**模糊了根内区分度**,恰好没收了 S2 在更"锐利"的 base 表示上
   本能取得的增益;
3. **S1 有全局精度税**:孤立层 gold@10 0.116→**0.065**(6/8 seeds 变差)。
   根级池化是训练时的**全局**改动,牺牲 per-dataset 精度换根一致性——
   对不需要根一致性的孤立数据集是纯伤害。

### 洞察:为什么 serving 侧(S2)胜过训练侧(S1)

sibling 信号**只适用于一部分查询**(82% sibling-rich,18% 孤立)。
- **S2 是外科手术式**:只在有兄弟处触发,孤立层跨 α 逐位不变、零副作用;
- **S1 是全局式**:sibling-rich 上小幅噪声增益,却对孤立层征收真实精度税,
  且与 S2 冗余后反而有害。

**对一个只适用于子集的信号,外科式交付(S2)必然优于全局式交付(S1)。**
这不推翻 sibling 通道(S1-alone 证明它真实),而是确认 **S2 是它的正确交付
机制**——v6 架构(serving 侧 sibling 融合)被反向验证。

**决定:S1 不进服务栈;S2(α=1)维持 sibling 通道的唯一交付。**
S3(兄弟辅助监督)若重访,须避开 S1 的两个坑:全局精度税 + 与 S2 冗余。

## 5. 复现

```bash
cd ModelLakeFishing  # 仓库根
# node-level(S1 + S1×S2,对照见 s2_sibling 基线)
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.s2_sibling \
    --s1 --seeds 0 1 2 3 4 5 6 7 --epochs 25
# root-aware(S1 vs L1L3b)
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.w1_dzero \
    --rows L1L3bS1 --seeds 0 1 2 3 4 5 6 7 --tag l1l3bs1_rootaware \
    --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_d0_v1.pt
```