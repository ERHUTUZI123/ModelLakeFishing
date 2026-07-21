# D1 特征重构可审计记录 —— x_m⁽⁰⁾′ = [e_name↓ ‖ e_size ‖ e_task]

**日期:** 2026-07-18
**上游方案:** `RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §5.3(新特征方案)/ §5.4(消融矩阵 F0–F4)
**范围:** 本文记录 D1 特征重构的**全部代码改动**与**每个 embedding 分量的数学定义**,
以及机制验证与消融运行方式。所有变体都是 encoder 视角的结构改造——
**图内的 `data['model'].x` 一个字节都没动**(图 sha256 在所有报告中保持不变)。

---

## 1. 特征形态总览(F0 → F4,one change per row)

| 行 | x_m⁽⁰⁾ 构成 | 总维 | 变化 |
|---|---|---|---|
| F0 | [e_name 64 ‖ e_desc 384 ‖ e_size 16 ‖ e_fam 16] | 480 | G2 复刻(对照) |
| F1 | [e_name 64 ‖ e_size 16 ‖ e_fam 16] | 96 | −e_desc(80% 弱信号) |
| F2 | [e_name 64 ‖ e_size 16] | 80 | −e_fam(lineage 边已承载家族) |
| F3 | [e_name 64 ‖ e_size 16 ‖ **e_task 16**] | 96 | +模型侧任务表(预期主涨点) |
| F4 | [**P·e_name 16** ‖ e_size 16 ‖ e_task 16] | **48** | name 结构性降维 64→16 |

> 勘误说明:方案原文写"新总维 64",按 §5.3 目标形态 [16‖16‖16] 实际为 **48**
> (64 是把 name 未投影的 F3 形态误算;不影响任何设计决策)。

## 2. 每个分量的数学定义(含冻结/可学习边界)

### 2.1 e_name(冻结,64 维)—— hash-平均词嵌入

对模型名 `m`(org + repo + 子词)分词得 tokens(t_1..t_k),
每个 token 经 MD5 哈希入桶:

```
b(t) = MD5(t) mod 10 000                       (HASH_BUCKETS = 10 000)
T ∈ R^{10000×64},  T_i ~ N(0, I_64) 后逐行 L2 归一化(seed = 42,固定)
e_name(m) = (1/k) Σ_j T[b(t_j)]               (k=0 时为零向量)
```

冻结:T 由 seed 决定性重建,不进参数组。微调模型的命名习惯把训练数据集写进名字
(`bert-finetuned-sst2`),因此 e_name 是当前最强的任务信号载体之一(§5.2 警告)。

### 2.2 e_desc(冻结,384 维;F1 起砍除)

`all-MiniLM-L6-v2` 对 model card 文本的句向量。审计结论(E6):HF model card
大量为模板文本,该分量占输入 80% 维度却高度互相似——弱信号淹没强信号的主体。
砍除方式 = encoder 内切片 `x[:, :64]`(name 半)/`x[:, 64:]`(desc 半)后弃用后者,
**图不重建**。切片边界 `name_dim=64` 来自 `xm0_meta['name_dim']`(图内自带)。

### 2.3 e_size(可学习,16 维)—— 半十进位对数桶

```
bucket(c) = 0                                  c 缺失/非正   (unknown 桶)
          = clamp(1 + ⌊(log10(c) − 5.0)/0.5⌋, 1, 14)   否则
NUM_SIZE_BUCKETS = (12.0 − 5.0)/0.5 + 1 = 15
E_size ∈ R^{15×16}(nn.Embedding,可学习)
```

id 0 学的是"缺失本身的先验"(设计特性)。常数是代码级契约,改动即需重训。

### 2.4 e_fam(可学习,16 维;F2 起砍除)

规则+动态推断的 185 族 vocab(Other=0,`FAMILY_MIN_COUNT=3` 折叠),
`E_fam ∈ R^{185×16}`。砍除理由:家族结构已由 lineage 边(is_base_of/rev)
在消息传递层承载,特征层是冗余通道;185 行长尾表的 rare rows 学不到东西。
`use_family=False` 时表不构建(参数直接消失,而非置零)。

### 2.5 e_task(新增,可学习,16 维)—— 模型侧任务表

**vocab 构建**(`d1_model_task_vocab.py`,离线,零网络):

```
task(m) = canon(pipeline_tag)                  首选(1938/2000 模型命中)
        = canon(mode{model-index task.type})   回退(57)
        = Other                                兜底(5 无 JSON)
canon(·):拼写/粒度折叠表(sentiment-analysis→text-classification,
         bitextmining→sentence-similarity 等 40 条规则)
计数 < TASK_MIN_COUNT(5) 的任务折入 Other      (FAMILY_MIN_COUNT 同构)
```

**结果 vocab = 9 行**(Other=0 钉死):text-classification 610 / sentence-similarity
592 / feature-extraction 340 / token-classification 301 / question-answering 70 /
translation 25 / fill-mask 19 / text-generation 17 / Other 26,覆盖率 98.7%。

```
E_task ∈ R^{9×16}(nn.Embedding,可学习)
x_m 拼接项 = E_task[task_id(m)]
```

**零样本规则(归纳性不破):** 新模型 pipeline_tag 缺失或不在 vocab → id 0(Other),
与 size 的 unknown 桶、family 的 Other **完全同构**(`learnable.new_model_task_id`)。
**身份凭证纪律:** `task_vocab.csv` 与 checkpoint 绑定保存/加载校验
(连续双射 + Other→0,否则拒绝——family vocab 的同一套 orphan-row 防线)。

### 2.6 P·e_name(F4)—— 冻结高斯随机投影的结构性降权

**为什么标量降权无效(必须写清的工程事实):** encoder 是 concat + Linear。
对任意固定标量 α,第一层 `W·(α·e_name) = (αW)·e_name` —— α 被 W 完全吸收,
训练后等价于没降权。**维度是唯一不可被线性层吸收的容量约束。**

```
P ∈ R^{64×16},  P_ij ~ N(0, 1/16)(seed = 42,决定性重建)
e_name↓ = e_name · P
```

- **Johnson–Lindenstrauss 缩放:** E‖xP‖² = ‖x‖²(P_ij 方差 1/proj_dim 保范数期望),
  测试实测比值∈(0.5, 1.6) 通过;
- **冻结身份:** P 以 `register_buffer` 注册——随 checkpoint 保存、随 `.to(device)`
  迁移、**不进参数组、无梯度**(测试断言 `named_parameters` 不含它);
- seed 记入 arch,同 seed 同维 → P 逐位可重建。

### 2.7 梯度边界(全变体不变量)

```
∂L/∂E_size, ∂L/∂E_fam, ∂L/∂E_task ≠ 0    (稀疏行梯度,batch 命中行才更新)
x_frozen(图内 .x):永不回写(非 Parameter);
P:buffer,无梯度;
F1/F4 下 desc 半列(x[:, 64:])完全脱离计算图(测试:∂out/∂x[:,64:] ≡ 0)。
```

## 3. 代码改动清单

| 文件 | 改动 |
|---|---|
| `stage1BuildTransferGraph/d1_model_task_vocab.py` | **新增**:e_task vocab 离线构建(§2.5),产物在 `artifacts/d1_features/`(task_vocab.csv / model_task_ids.csv / d1_task_report.json) |
| `stage1BuildTransferGraph/dataset_embed/model_node_encoder.py` | `ModelNodeEncoder` 扩展 7 个 kwargs(name_dim/use_desc/use_family/name_proj_dim/name_proj_seed/num_model_tasks/task_dim);forward 增加可选 `task_id`;**默认参数下与旧行为逐位一致**(测试保证) |
| `stage2TrainGraphSAGE/model.py` | `HeteroGraphSAGE` 透传上述 kwargs;`encode_nodes` 从 model 节点存储取 `task_id`(与 size/family id 同一 ride 机制,loader 自动切片) |
| `stage2TrainGraphSAGE/learnable.py` | `_arch_of` 记录变体字段(family 缺席记 0);`save_checkpoint` 新增 `model_task_vocab` 绑定 + 校验 + sidecar CSV;`load_checkpoint` 按 arch 重建(**旧 checkpoint 默认值兼容,已测**);`new_model_task_id`/`encode_new_model` 零样本路径;`snapshot_weights`/`embedding_health` 按实际存在的表工作 |
| `stage2TrainGraphSAGE/d1_features.py` | **新增**:`attach_model_task_ids(data, umi)` 运行时附加 task_id 列(mappedID + 模型名**双重对齐校验**,错图 CSV 直接拒绝)+ vocab 加载 |
| `stage2TrainGraphSAGE/ablation.py` | `train_eval_one` 读 cfg 的 `drop_desc/drop_family/name_proj_dim/use_model_task` 四个 flag 组装 encoder kwargs(缺省 = legacy) |
| `stage2TrainGraphSAGE/top1_baselines.py` | `run_configs` 在 split 之前按需 attach task_id;每 split 记录 `train_diag`(tau/mean_cos/z_m_pr/z_d_pr,G-D1 验收需要) |
| `stage2TrainGraphSAGE/d1_fphase.py` | **新增**:F0–F4 行定义(G2 配置全冻结,只动特征)+ G-phase 晋升门 + z_m PR 不降的 G-D1 附加门 |
| `stage2TrainGraphSAGE/tests/test_d1_features.py` | **新增**:10 项机制测试(下节) |

## 4. 验证记录

**新增测试 10/10 通过**(`pytest tests/test_d1_features.py`):

1. legacy 默认路径与旧实现逐位一致(frozen 块直通);
2. F1–F4 out_dim 精确(96/80/96/48);
3. desc 切片正确性(扰动 desc 半不影响输出);
4. P 同 seed 决定性、异 seed 不同、buffer 非参数、JL 范数比通过;
5. 梯度边界:三表有梯度、P 无梯度、desc 半零梯度;
6. 有 task 表而缺 task_id → 显式报错;
7. 变体缺 name_dim → 显式报错;
8. 真图全模型 roundtrip:attach → forward → save/load 输出 allclose,vocab sidecar 落盘;坏 vocab(无 Other/非双射)拒存;有表无 vocab 拒存;
9. 错图 CSV(mappedID 不一致)attach 拒绝;
10. **旧 G2 checkpoint 原样加载**(use_desc/use_family 默认、无 task 表、out_dim 480)。

**存量测试回归:** stage2 的 phase0/phase1/phase4-5 脚本式测试全部 OK。

## 5. 复现

```bash
# 1) e_task vocab(离线,~1 分钟)
cd ModelLakeFishing/stage1BuildTransferGraph
../.venv/Scripts/python.exe d1_model_task_vocab.py        # artifacts/d1_features/

# 2) 机制测试
cd ../stage2TrainGraphSAGE
../.venv/Scripts/python.exe -m pytest tests/test_d1_features.py -q   # 10 passed

# 3) F0–F4 消融(3 split × 5 行,清洁五指标 + z_m PR 门)
cd ../..   # 仓库根
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage2TrainGraphSAGE.d1_fphase \
    --rows F0 F1 F2 F3 F4
# 产物: stage2TrainGraphSAGE/artifacts/ablation/top1/d1_fphase/{F0..F4}.json,
#        F_report.{json,md}
```

vocab 覆盖数字复核:

```python
import json
r = json.load(open(r'ModelLakeFishing\stage1BuildTransferGraph\artifacts\d1_features\d1_task_report.json'))
assert r['num_model_tasks'] == 9 and r['n_models'] == 2000
assert r['coverage_by_source'] == {'pipeline_tag': 1938, 'model_index': 57, 'none': 5}
assert abs(r['share_other'] - 0.013) < 1e-3
```

## 6. F0–F4 消融结果(2026-07-18,15 次训练完成)

晋升门 = clean 五指标规则(gold@10 全 split 升或 paired CI>0,hit@1 回退 ≤0.02,
regret 恶化 ≤0.005)∧ z_m participation ratio 不降。
产物:`artifacts/ablation/top1/d1_fphase/{F0..F4}.json, F_report.{json,md}`。

| 行 | observed hit@1 | top3 hit@1 | regret@1 | gold@10 | z_m PR | 判定 |
|---|---|---|---|---|---|---|
| F0(G2 复刻) | **0.400±0.062** | 0.652±0.025 | 0.051±0.009 | 0.046±0.035 | 3.1 | 对照 |
| F1(−desc) | 0.344±0.086 | 0.621±0.061 | 0.054±0.006 | **0.053±0.035** | 3.6 | REJECT |
| F2(−desc−fam) | 0.320±0.030 | 0.586±0.031 | 0.081±0.010 | 0.054±0.039 | 2.9 | REJECT |
| F3(F2+e_task) | 0.310±0.058 | 0.548±0.067 | 0.067±0.020 | **0.000±0.000** | 3.4 | REJECT |
| F4(F3+name↓16) | 0.321±0.009 | 0.597±0.005 | 0.059±0.007 | 0.036±0.011 | 3.2 | REJECT |

**F0 复刻校验:** 与历史 G2 记录一致(gold@10 0.046±0.035,hit1 0.40)——harness 同底可信。

**诚实读数(不是全盘失败,但假设被部分证伪):**

1. **"desc 是噪声主体"只对了一半。** F1 的 gold@10 确实微升(0.046→0.053)
   ——砍掉 384 维后 gold 检索没有变差;但 observed hit@1 掉了 0.056(超 0.02
   容忍),说明 **desc 对带标签候选内部的排序有真实贡献**(容量或残余语义),
   不是纯噪声。方案 §5.3.1 的预案(desc 保留但降到 32 维投影)成为首选后续行。
2. **F3 的 gold@10 = 0.000(三 split 一致)是最重要的负结果。** e_task 只有
   9 个离散值,610 个 text-classification 模型共享同一行 16 维向量;在砍掉
   desc+fam 之后,同任务模型的输入特征只剩 name+size 可区分,e_task 反而把
   任务簇内部**压得更同质**——gold(NV-Embed 类特化模型)被彻底挤出 top-10。
   这与 CLAUDE.md"over-smoothing near hubs"警告同构:**粗粒度共享特征在特征
   稀薄时是塌缩加速器**。e_task 要么需要更细的任务粒度,要么必须与更强的
   区分性特征(如保留的 desc↓32)配对。
3. **z_m PR 全程 3±0.5**——嵌入空间内在维度没被任何变体实质改变,
   几何瓶颈仍在(与 diverse-zoo 诊断一致:病根更多在 z_d 条件化)。
4. 门禁按设计工作:多样性/维度变化没有换来相关性 → 拒收(CSLS 教训条款生效)。

**决策留白(待定,不自动执行):** 候选后续行 F1b = desc 冻结随机投影 384→32
(保残值、去容量霸权);F3b = e_task 换 task-配对门控或与 desc↓32 组合。
是否开跑由 one-change-per-row 纪律与人工决定。

## 7. 与 D0/D2 的衔接

- e_task vocab 构建器直接可跑在 D0 合并湖上(`--graph` 参数 + d0_lake_cache 已在
  JSON 检索路径里);D0 新图建成后重跑一次即得 10K 版 task ids。
- D2 的 P1 剪枝"accuracy × 任务匹配"打分依赖本文的 e_task/task_id 列——
  落地后即可启动 P1 任务加权版。
- 胜出 F 行经不变的 Stage-3 管线(export → build → query)重索引即上线,
  fidelity gate 照旧。
