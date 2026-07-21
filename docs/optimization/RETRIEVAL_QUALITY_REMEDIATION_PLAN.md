# 检索质量整改方案 v6 —— 按根结构分而治之:兑现 sibling 通道

**日期:** 2026-07-19(v6,见本节;v5/v4/v3 全文保留在后续章节,证据与纪律由 v6 继承)
**v5/v4/v3 日期:** 07-19 / 07-19 / 07-18;**v2:** 07-11;v1 为 07-10 英文版

---

## 0. v6 —— content 死路之后:真正的 headroom 是 sibling,不是特征

### 0.1 v5 收官后的判断转折

v5 排期已执行到 P1×P3 组合终裁(审计 `v5/P0_DIAGNOSIS.md`、`v5/P4_GOLDGAP_METRIC.md`、
`v5/P1_ECONTENT_FEATURE.md`)。结论链条:
**信息饥饿被证实(P0:O-task 0.080 < 现任 0.100)→ 内容通道被证伪
(P1:21% 覆盖,8-seed 分层 REJECT,承诺仅达成 7%)。**

v6 的核心判断:**v5 押错了通道。** P0 早已把答案摆在桌上——
**O-sibling = 0.563**(知道测试根其余数据集的标签时的上界),是 O-task(0.080)
和现任(0.100)的 5.6× 头顶。v5 试图用"数据集内容"去够这个上界,但内容
覆盖 21% 且与元数据冗余。**O-sibling 从定义上就不是内容信号——它是
"同根兄弟数据集的已知最优模型"这个 sibling-label 信号。** v6 直接兑现它。

### 0.2 决定性的结构事实(E17–E19)

**E17 —— 内容通道适用域太窄且冗余(v5 终裁,已证伪)。**
e_content 覆盖 21%(bitext 族 6.7%),79% 节点内容视图为零;有内容节点上
MiniLM 样本文本对 gold 区分未超出 e_card 元数据。8-seed:has_content 层
gold@10 名义 +0.029 但 per-seed 仅 4/8、gold-gap 反向 → 噪声。**内容不进服务栈。**

**E18 —— sibling 通道覆盖 82% 的 gold,且正是内容覆盖不到的那部分。**
图结构实测(复现见 §0.6):

| 量 | 值 |
|---|---|
| 数据集节点在 sibling-rich(多节点)根 | **78.5%**(1,149/1,463) |
| **gold 节点在 sibling-rich 根** | **81.7%**(503/616) |
| 多节点根数 | 107(25.4% 根,却含 78.5% 节点——bitext/massive 家族高度集中) |

**内容与 sibling 覆盖恰好互补(用户早在 P1 前就问过的"两条腿"):**
bitext 家族(54% 节点、内容覆盖 7%)是**兄弟最密、内容最缺**的——
它们的正解从来是 sibling-label,不是内容。v5 把力气花在了错的那条腿上。

**E19 —— 问题按根类型分叉,两类天花板不同。**
- **sibling-rich 根(82% gold):** 头顶 = O-sibling 0.563,**当前完全未兑现**
  (现任 z_d·z_m 没有任何机制利用"同根兄弟的已知最优")→ v6 主战场;
- **孤立/单节点根(18% gold):** 内容已证伪,图信息下接近天花板 →
  诚实接受,不再投入(除非 §0.5-P3' 的宽 seed 扩根真正落地)。

### 0.3 v6 工作包

**S 轨 —— sibling 通道兑现(主战场)**

| 行 | 内容 | 为什么 |
|---|---|---|
| **S2(headline,可部署,零重训)** | **serving-time sibling-prior 融合**:查询数据集 D 到达时,取 D 在**湖内**的同根兄弟(有 trained_on 标签的)的已知高分模型,作为先验与 MIPS 分数 `z_d·z_m` 融合。**这是 O-sibling 在服务侧的合法实现**——用的是湖里已有的标签,不是测试泄漏 | O-sibling 0.563 的直接兑现;零训练、只在 serving 层加一步;覆盖 82% gold |
| S1(训练侧) | 根级正样本池化(v5 的 P5 升为主行):训练时同根共享 top-frac 正样本,让 z_d 学"根不变的模型偏好" | 让 sibling 信号进入表示,而非只在 serving 融合;S2 的训练侧对偶 |
| S3(条件) | 若 S1/S2 显示兄弟间偏好高度一致:把同根兄弟的 trained_on 边作为**辅助监督**(D 的候选继承兄弟的相对序) | 仅 S1 诊断支持才开 |

**T 轨 —— task 元数据未用足(便宜,平行)**

| 行 | 内容 | 为什么 |
|---|---|---|
| P2b | **serving-side 分数混融**:z_d·z_m 与 O-task 分数(同任务均值)线性混合,零训练 | P0:predictable 层(9% 根,O-task 0.87)现任仅 0.31——task 元数据未用足;最便宜的一行 |

**E 轨 —— 扩根的诚实再定位**

- **P3'(可选,重投入):** v5 证明放宽门槛只 +40 根(绑定约束是 seed 扫描广度)。
  真扩根需**更宽 seed 重扫**(新类目/benchmark 搜索/lineage 爬取,目标 +数百根)。
  **默认不做**——仅当 S/T 轨兑现后仍需要更多孤立根覆盖时才投入。

### 0.4 数值目标(按根分层,替代 v5 的单一承诺)

v5 的单一承诺(root gold@10 0.22–0.29)已被证伪不可由内容达成。v6 按 E19 分层:

- **sibling-rich 层(82% gold):** 目标 = 关闭现任→O-sibling(0.10→0.563)差距的
  **35–55%**,即该层 root gold@10 达 **0.26–0.35**、gold-gap@10 ≥ 0.50;
  由 S1+S2 在 8 seeds 上验收(分 sibling-rich / isolated 两层报告);
- **isolated 层(18% gold):** 诚实接受接近天花板,不设涨点目标,只报不退步。

### 0.5 排期

```
第 1 步   S2 serving 融合(零训练,先行,最快见效)+ P2b 混融——同一 serving 评测框架
          结构事实 E18 复算脚本(sibling 覆盖,已备)
第 2 步   S1 根级正样本池化(训练侧,8 seeds,分层验收)
第 3 步   S1×S2 组合 8-seed 终裁,对齐 §0.4 分层目标;胜者 Stage-3 重索引
          (S2 是 serving 层,可能需要索引侧 sibling 映射)
之后      S3(条件)、P3'(仅在孤立根成为瓶颈且值得投入时)、47K
```

### 0.7 S2 执行状态(2026-07-19,详见 `v6/S2_SIBLING_FUSION.md`)—— **采纳**

S2(serving-time sibling-prior 融合,零重训)已 8-seed 落地。**v3→v6 全程自 L1
以来第一个干净、显著、8/8 一致的正结果:** node-level 切分(cold-D/warm-siblings,
对应 E19 的 sibling-rich 场景)下,has_sibling 层 gold@10 **0.162→0.266(+64%,
8/8 seeds,paired CI [+0.040,+0.089] 排除 0)**、gold@1 3.2×;**no_sibling 层跨所有
α/seed 逐位不变**(合法性边界绝对可靠,零副作用)。**落入 §0.4 sibling-rich 目标
带 0.26–0.35(关闭差距 36%)**;gold-gap@10 0.330 未达 0.50(部分,靠 S3)。
采纳 α=1.0 进 serving 栈,针对 sibling-rich 查询(82% gold),孤立根自动不触发。
合法性:boost 只用训练可见兄弟节点、显式排除 D 自身(≠ O-sibling oracle 泄漏)。

### 0.8 S1 执行状态(2026-07-19,详见 `v6/S1_ROOT_POOLING.md`)—— **不采纳,但反向验证 S2**

S1(根级正样本池化,训练侧对偶)8-seed 落地。**REJECT:** node-level 下
S1-alone(has_sibling 0.162→0.195 仅 4/8 seeds,噪声);**S1×S2(0.232)< 单独
S2(0.266),6/8 seeds S1 拖累**(S1 在表示层模糊根内区分度,与 S2 冗余后有害);
**孤立层 S1 有全局精度税(0.116→0.065,6/8 变差)**;root-aware 中性偏负
(gold@10 0.100→0.095、gold@1 降)。**洞察:sibling 信号只适用 82% 子集,
外科式交付(S2 只在有兄弟处触发、孤立层零副作用)必然优于全局式(S1 训练时
全局池化,征税孤立层)。S1-alone 证明兄弟偏好真能烧进 z_d,但确认 S2 是正确
交付机制——v6 架构被反向验证。** S1 不进服务栈,S2(α=1)维持 sibling 通道
唯一交付。新增 `losses.pool_membership_by_root`。

### 0.9 P2b 执行状态(2026-07-19,详见 `v6/P2b_TASK_FUSION.md`)—— **采纳,与 S2 叠加,补全孤立层**

P2b(serving-time task-prior 融合,零重训,与 S2 正交)8-seed 落地采纳。
决定性:**(1) P2b 兑现了 S2 够不到的孤立层**——no_sibling 层 MIPS 0.121 →
P2b(β=1)**0.180**(6/8 seeds,配对 CI [+0.025,+0.100] 排除 0),S2 该层 0.121
不变(无兄弟不触发)。**这正是 v5 内容通道证伪失败的 18% 孤立根,由 task
通道补上。(2) S2+P2b 叠加**(不同于 S1 替代):all 层 sibtask 0.219 > sib
0.188(7/8 seeds),互补(sibling 主导 sibling-rich、task 独撑孤立层)。
**统一律:serving 侧外科式加法融合的合法湖标签先验可叠加,训练侧全局池化不行。**
**采纳统一重排 `fused = minmax(MIPS) + α·sibling_boost + β·task_boost`(α=β=1),
一套逻辑覆盖两类根(boost 各自在不适用处自动归零),完成 E19 分叉可部署闭环:
sibling-rich(82%)S2 主导、孤立(18%)P2b 兑现。** 新增 `p2b_task.py`。

### 0.10 serving 层集成执行状态(2026-07-19,详见 `v6/SERVING_INTEGRATION.md`)—— **已接入,零重索引**

统一 sibling+task 重排已接入已上线 `d0_L1L3b` 服务栈:`build_prior_sidecar.py`
(trained_on 邻接 + root/task 映射,mappedID 对齐 export 铁律)+ `serving_rerank.py`
(`fused = minmax(MIPS) + α·sibling + β·task`,α=β=1,`_prior` 排除 D 自身)。
端到端验证:sibling-rich 查询两通道生效、孤立查询 S2 静默 P2b 独撑(E19 设计);
**合法性断言全过**(孤立查询 sibling_boost 逐元素零、仅与 D 有边的模型被排除、
index.bin 全程未写)。**z_m/z_d/HNSW 索引零改动——无需重索引**,只多一个 sidecar
+ 检索后一步重排。晋升配置 L1L3b 及索引不变。剩:α/β 暴露为查询参数、
retrieval_service 接线供 Stage-4、S3(条件,须避 S1 两坑)。

### 0.6 纪律(继承 v5 + 新增)

1. **8 seeds 起步、按根类型分层报告**(sibling-rich vs isolated)——v6 的判读
   必须分层,全局均值会混淆两类天花板;
2. gold@10(球门柱)+ gold-gap@10(δ=0.01 产品口径)双主指标(P4 已落地);
3. **S2 的合法性边界**:serving 融合只能用**训练可见/湖内已有**的兄弟标签,
   绝不用测试根自身的held-out 标签(那才是 O-sibling 的泄漏部分)——
   这条是 S2 与 oracle 的分界,实现时必须断言;
4. 负结果字典继承:content(E17)、F/Z/L4/P2-cap、CSLS、单纯放宽扩根门槛(E18)。

结构事实复现:

```python
import torch, pandas as pd
from collections import Counter
p = torch.load(r'ModelLakeFishing\stage1BuildTransferGraph\hgraph_d0_v1.pt',
               map_location='cpu', weights_only=False)
udi = p['unique_dataset_id'].sort_values('mappedID')
pool = pd.read_csv(r'ModelLakeFishing\stage1BuildTransferGraph\artifacts\d0_lake\d0_dataset_pool.csv')
gold = set(pool[pool['gold_evaluable']]['dataset_node'])
roots = udi['root'].tolist(); nodes = udi['dataset'].tolist()
rc = Counter(roots); multi = {r for r,c in rc.items() if c>=2}
gn = [n for n in nodes if n in gold]
assert round(sum(r in multi for r in roots)/len(nodes), 3) == 0.785
assert round(sum(roots[nodes.index(n)] in multi for n in gn)/len(gn), 3) == 0.817
```

---

# 检索质量整改方案 v5 —— 从分布修正到信息补给(已执行,由 v6 收束)

**日期:** 2026-07-19(v5;排期已执行至 P1×P3 终裁,状态注见各节)
**v4/v3 日期:** 2026-07-19 / 07-18;**v2:** 07-11;v1 为 07-10 英文版

---
v5 已完成 P0 诊断 和 P3+P1 采集。剩余待执行:

主线(训练环节)

P4 落地:把 gold-gap@10(δ=0.01)写进评测 harness 作共同主指标——目前只定了参数,还没进报表代码
e_content 特征化:MiniLM 编码 P1 的样本文本 → e_content 视图 → 拼入 xd0
P1×P3 组合建图:D0.5 湖(461 根/10,234 模型)+ 内容视图,一次建成
8-seed 分层验收:L1L3b+content 消融,只在 has_content 分层(164 gold 节点)看增量
组合行 8-seed 终裁:对齐 §0.6 承诺(root gold@10 0.22–0.29 / gap@10 ≥0.45)
胜者 → Stage-3 重索引(管线零改动)
平行/条件
7. P2b 分数混融(serving 侧零训练,针对 predictable 层 0.31 vs 0.87 缺口)——可随时插入
8. P2(top-heavy 精修)、P5(根级池化)——条件行,仅特定证据支持才开

已明确推迟(远期)

更宽的 seed 重扫(P3 诊断出的真正根扩容,需更大网络作业)
47K 扩展(pyg-lib / z 缓存 / alibi 分位复核)
gpt-neo 探针视图回补(仅当 z_d 条件化再成瓶颈)
下一个自然执行单元是 步骤 1–4(P4 + 特征化 + 组合建图 + 分层消融),其中步骤 2-3 是最大工程量。

## 0. v5(2026-07-19)—— gold@10 = 0.10 之后:缺的不再是校正,是信息

### 0.1 v4 收官后的局面与本次修订的性质

v4 排期全部执行完毕(审计:`v4/W1_D0_AUDIT.md`、`v4/W3_STAGE3_AUDIT.md`):
L1L3b 晋升、Stage-3 索引 fidelity 1.000、P2 卫生行 8-seed 中性。
**最终数字:root gold@10 = 0.100±0.058(全湖 9,491 分母)——用户判定不足。**

v5 的核心判断:**v3/v4 修的是监督分布的"偏"(盲区、度数偏置),这条线已经
拧到头**——logQ 全湖(2.4×)、task 元数据、n_neg 256 之后,IPW 正样本证伪、
硬负样本中性、消息图卫生中性(8-seed)、表示手术全线证伪。0.100 之上的缺口
性质不同:**root-aware 切分下,测试根的全部标签不可见——模型必须纯靠
"数据集长什么样 + 模型在别的根上的表现"做跨根泛化。而查询侧现在只有
一句元数据描述串。这不是分布偏差,是信息饥饿。**

### 0.2 新证据与判断依据(E14–E16)

**E14 —— E7/E13 的适用域到期(重要的自我修正)。**
"表示不是杠杆"的证据全部来自 2K/节点级切分——那时测试数据集的兄弟节点
在训练中可见,特征信息被亲缘泄漏冗余化,砍掉当然无损。D0 root-aware 下
测试根完全不可见,**查询特征是唯一的跨根通道**。同时:z_d PR 在所有目标
变体下都停在 ~3.4(E11 已证:目标能榨的都榨了),而输入侧只有 17 类 task id
+ 一句 MiniLM 描述串——**当输入信息只有这么多,任何目标函数都造不出更高的
条件化**。这构成 v4 §0.2-2 重访门槛条款要求的证明性诊断:"区分 421 个根的
查询几何是目标函数不可能自发达成的,因为所需信息不在图里"。
**据此,v5 批准唯一一项表示侧重开:数据集内容特征(P1);F/Z 系维持关闭。**

**E15 —— 跨 seed 根族方差指示信息不均。** root gold@10 per-seed 0.014–0.19:
有的根族(任务典型、训练近亲多)靠现有信息可预测,有的完全不可。
开工前先诊断两类占比(P0)。

**E16 —— gold 的"近平局"结构未被度量(P0 实测)。** 9,491 湖里 gold 常有
ε 级近平局(标签噪声 + median 折叠下"唯一最优"部分是任意的)。gold@10 把
"top-10 有一个与 gold 差 0.5% 的模型"记为 miss——低估真实服务效用,
也让优化目标偏离产品目标("找到一个够好的模型")。

### 0.3 v5 工作包

**P0 —— 诊断漏斗(先于一切,零训练,~半天)**

对 8 个 root-aware seed 的测试根统一计算:

| 量 | 定义 | 决策含义 |
|---|---|---|
| O-static | 全局流行度/均值榜的 root gold@10 | 条件化真实贡献的底线 |
| O-task | 只保留 task 画像匹配模型、按同任务均值排序 | **关键分叉:** ≫0.100 → 现有信息未榨干,目标/结构仍有余量(先 P2);≈0.100 → 信息饥饿证实,P1/P3 主攻 |
| O-sibling | 用测试根自身标签(故意泄漏)的上界 | 可达上限;与 O-task 的差 = 内容信息的价值上界 |
| cold-gold 率 | gold 模型的全部标签都在测试根上(z_m 无监督信号)的占比 | 无内容特征则结构性不可达的部分 |
| 近平局分布 | 每根 gold 与第 2..10 名标签值 gap 的分布 | P4 的 δ 取值依据 |
| 根可达性分层 | 按 O-task 把根分"可预测/不可预测"两层 | P1 验收看"不可预测层"的增量 |

**P1 —— 查询侧内容特征回补(v5 唯一的表示侧重开,E14 授权)**

- 采集:HF datasets-server rows API 拉每节点 ~100 条样本文本(1,463 节点,
  有界网络,断点续跑,复用 d0_online_harvest 缓存纪律;config 覆盖率 P0 顺带盘点);
- 特征:样本文本经 encoder(先 MiniLM 验方向,gpt-neo 探针为升级项)
  → e_content 视图拼入 xd0(建图器 view_dims 结构已留位);
- 行:L1L3b + e_content,**8 seeds 起步**;验收 = root gold@10 配对 CI
  + "不可预测层"分层增量;
- 内容是数据侧信息,与标签无关,零泄漏问题。

**P2 —— 目标的 top-heavy 精修(一行,小)**

正样本按**精度差距**加权:within-dataset 里 acc 越接近该数据集最优,权重越大。
与已证伪的 L4 严格区分——L4 按度数折减(流行度不是监督),这里按标签值加权
(监督的强度本来就是监督的一部分)。效果指向:容量集中于"认出每个数据集的
头部",直接对准 gold@K。

**P3 —— 监督面扩容 D0.5(平行开跑,机器已证)**

v4 湖是 Phase-B 保守参数的产物:keep 门槛"邻域 ≥2 eval 模型"只留
530/5,637 候选、每数据集 cap 40。放宽一轮:门槛 ≥1(+model-index 交叉引用
救回)、cap 40→100、d0_online_harvest 增量复跑(全缓存,新增部分才花网络)。
目标:训练根 421 → ~800+、监督边 55K → 100K+。
**跨根泛化的第一营养是更多的根。**

**P4 —— 评测语义升级(与 P0 同批落地)**

新增共同主指标 **gold-gap@10**:top-10 内最优模型与 gold 的标签差 ≤ δ 即计中
(δ 由 P0 近平局分布实测定,预期 0.01–0.02);gold@10 保留不动
(不移动球门柱,加一根真实的);median gold rank 进常设报表。
目标从"找到那一个 gold"校正为"找到一个够好的"——这是产品语义而非放水:
δ 从标签噪声结构里测出,不是拍的。

**P5(条件行)** —— 仅当 P0 显示同根内偏好高度一致:训练加根级正样本池化
(train 根内共享 top-frac),学根不变式。

### 0.4 纪律更新(v4 的学费)

1. **8 seeds 起步,3 seeds 永不裁决**(W3 假信号:3-seed 全升 + CI 排除 0
   仍被 8-seed 均值回归推翻);
2. 表示侧重开仅限 P1(E14 授权);F/Z/L4/P2-cap 维持负结果字典;
3. 数值目标在 P0 oracle 出来后定,承诺形式为"关闭 O-task 与现状差距的 X%、
   cold-gold 层由 P1 打开 Y 个点",不承诺裸数字;
4. Stage-3 契约不变;功能等价冻结门(W3 审计 §1.2)为既定先例。

### 0.5 排期

```
第 1 步   P0 诊断漏斗 + P4 评测升级(同批产物,零训练);P3 补采后台启动
第 2 步   按 P0 分叉:O-task ≫ 0.100 → 先 P2 行;否则直上 P1 采样+特征行
第 3 步   D0.5 建图(P3 产物)+ 现任配置 8-seed 重基线
第 4 步   P1×P3 组合行 8-seed 终裁 → 胜者 Stage-3 重索引
之后      P5(条件)、47K(pyg-lib/z 缓存/alibi 分位)
```

### 0.6 P0 执行状态与分叉裁决(2026-07-19,详见 `v5/P0_DIAGNOSIS.md`)

P0 已完成(零训练,8 seeds × 1,292 数据集)。裁决:

1. **分叉走 P1/P3 主线。** O-task oracle@10 = 0.080 **<** L1L3b 0.100——
   GNN 已榨干任务画像信息,"先精修目标"分支关闭;P2 降为条件行。
2. **信息价值定量:O-sibling@10 = 0.563**(知道根标签的泄漏上界)——
   现状与上界的 4.6× 差距全部是"根信息"的价值,P1/P3 的供给对象。
   **cold-gold 率 27.9%**(gold 零训练边)为边法的结构上限背书。
3. **E16 证实并落地 P4:δ = 0.01**(70% 数据集第 2 名在 gold−0.01 内,
   中位差 0.0017、P25 精确并列)。现任 **root gold-gap@10 ≈ 0.35**
   (s0 实测)——产品语义下效用被严格 gold 低估 ~3 倍。
4. 新增 **P2b 分数混融**(serving 侧零训练)入队:predictable 层
   (O-task ≥0.5 的 9% 根)上 L1L3b 只拿 0.31 vs oracle 0.87,
   现任模型未用足任务画像最强段。
5. **v5 数值承诺(依 §0.4-3 格式):关闭 0.100→0.563 差距的 25–40%,
   即 root gold@10 达 0.22–0.29 / root gold-gap@10 ≥ 0.45,
   由 P1×P3 组合行 8-seed 验收。**
6. P1 实况约束:datasets-server 覆盖 52/100——内容特征为部分覆盖增强,
   验收按有/无样本分层。

### 0.7 P3+P1 采集执行状态(2026-07-19,详见 `v5/P3_P1_HARVEST.md`)

两条有界网络管线已启动(全缓存断点续跑,仅采元数据/样本文本):

- **P3(D0.5 扩容,已完成):** min_keep 2→1 + cap 40→100,+908 模型。D0.5 建图
  口径:strict 模型 9,491→**10,234**、训练根 421→**461**、监督边 54,795→**55,140**。
  **幅度远低于方案"根→800+/边→100K+"的预期——诚实诊断:绑定约束是已缓存 seed
  扫描的广度(候选池 12,153 已定),放宽门槛只救回弱监督数据集,cap 加深而非
  拓宽。** 真正的根扩容需更宽 seed 重扫(推迟)。执行决策:+40 根不值单独重基线
  (必落 v4 噪声内),**并入 P1×P3 组合图一次验证**(§0.5 步 3 独立重基线取消)。
- **P1(内容特征,已完成):** 1,463 节点,**has_content 312(覆盖 21.3%)**。
  低于探针 52% 因图有 53.8% 是 bitext 兄弟节点(datasets-server 供文本仅 6.7%);
  但分层有利:**非 bitext 覆盖 38.3%、gold 节点 26.6%(164 个)**。
  与 P3/sibling 的标签腿**覆盖互补**(bitext 靠兄弟标签,孤根靠内容)。
  **验收纪律强化:P1 消融只在 has_content 分层看增量**(no-content 回退元数据串
  作对照),全局均值被 79% 无内容节点稀释,不作主判据。特征化+建图为独立离线步。

### 0.9 e_content 特征化 + 组合建图执行状态(2026-07-19,详见 `v5/P1_ECONTENT_FEATURE.md`)

P1 样本文本经 MiniLM 编码为 **e_content 视图**拼入 xd0(458→842);无内容节点
e_content 置零、元数据视图保留(= 回退纯元数据),has_content 标志进 e_stats +
挂节点存储供分层消融。P3 D0.5 湖同时并入(§0.7 决策)。**组合图
`hgraph_d0_v1_content.pt`(sha dea91542)**:models 10,234、roots 461、
xd0 842、content 覆盖全局 20.6%/gold 26.6%;v4 `hgraph_d0_v1.pt` 不动,两图并存。
`d0_build_graph.py --content`(默认 OFF 复现 v4);`w1_dzero --graph` 已就绪。

**8-seed 分层终裁(2026-07-19):P1 内容 REJECT,§0.6 承诺未达成。**
干净隔离(内容图 vs 同湖 e_content 置零控制图,同 seed 同测试集,16 训练):
has_content 层 gold@10 名义 +0.029 但 **per-seed 仅 4/8 上升 + gold-gap@10 反向**
= 噪声;no_content 对照层内容反而略低(徒增全零参数)。承诺检查:内容图
root gold@10 = **0.134**(目标 0.22–0.29,**仅关闭差距 7%**)、gold-gap@10 0.211
(目标 ≥0.45)。诊断:覆盖 21% → 内容视图 79% 是零;有内容节点上 MiniLM 样本
文本对 gold 区分未超出 e_card 元数据(冗余);功效不足。**L1L3b @ hgraph_d0_v1.pt
维持晋升配置(索引仍服务),P1/P3 不进服务栈。** O-sibling 0.563 头顶空间在
21% 覆盖下无法由内容兑现——真正 headroom 在 sibling-label 通道(P5 根级池化,
bitext 家族正解)或覆盖远高于 21% 的采集(非 datasets-server)。

### 0.8 P4 执行状态(2026-07-19,详见 `v5/P4_GOLDGAP_METRIC.md`)

gold-gap@K 已落地评测 harness(`top1_eval.py` 一处源头,全链继承),δ=0.01
(P0 实测)。gold-gap@K = "top-K 内有候选 acc ≥ gold−δ";数学上 gap_rank ≤
gold_rank,是 gold@K 的严格松弛,**gold@K 球门柱不动、gate 仍用严格 gold@10**。
`w1_dzero` 全报表(root_macro/flat/pooled/MD)加 gold-gap@10 共同主指标。
机制测试全过;s0 冻结 ckpt 过新 harness 复现 P0 spot-check(root 0.351/flat 0.551),
**中位 gold rank 36 → 中位 gap rank 4**(近平局效用被严格口径低估的量化)。
现任锚点 root gold-gap@10 ≈ 0.35。

---

# 检索质量整改方案 v4 —— 单一战场收束:监督分布为纲,D0 为裁判(已执行完毕)

**日期:** 2026-07-19(v4;排期已全部执行,状态注见各节)
**v3 日期:** 2026-07-18;**v2 日期:** 2026-07-11;v1 为 2026-07-10 英文版

---

## 0. v4(2026-07-19)—— L1L3 之上:表示是目标的影子

v3 的 L/Z 双轨已在 2026-07-18/19 全部执行完毕(审计:`v3/L_TRACK_AUDIT.md`、
`v3/Z_TRACK_AUDIT.md`)。v4 不是又一次转向——它是把双轨的实验判决**收束成一条
主线**:L1L3 冻结为唯一候选,表示侧干预全面冻结,余下的问题只有两个
——**权衡轴的真伪**与**规模上的成立性**——而它们共用同一个裁判:D0。

### 0.1 新证据(E11–E13,全部可复现于 v3_lphase/v3_zphase 产物)

**E11 —— 目标函数顺手完成了 Z 轨的使命(v4 的基石证据)。**
z_d participation ratio(条件化的直接度量,v2 诊断的"内在维度 ≈ 2"):

| 配置 | z_d PR(3 splits) | 均值 |
|---|---|---|
| Z0(G2,无干预) | 2.46 / 2.83 / 1.87 | 2.39 |
| Z1(手工投影 128) | 1.76 / 2.37 / 1.17 | **1.77 ↓** |
| **L1L3(纯目标改动)** | **3.06 / 3.36 / 3.51** | **3.31 ↑,三 split 全升** |

L1L3 **没有改任何数据集侧架构**,z_d 条件化 +40%;Z1 专门为此设计的架构
干预反而 −26%。结论只有一种读法:**z_d 塌缩不是架构问题,是目标问题——
目标给了 z_d 区分数据集的理由(不同数据集要抑制不同的 hub 集合),几何就
自己展开了。表示是目标的影子。**

**E12 —— 权衡轴在 τ 层面是真实的,但其代价大小未裁决。**
组内精排 tau_macro:Z0 = 0.376,L1L3 = 0.278——lake 目标把容量从"带标签
候选内部的细排序"重新分配给"全局 gold/hub 分离"。这解释了 hit@1 的名义回退
(0.400→0.338);但 hit@1 的 bootstrap CI 全部跨 0(n=61 功效不足,E10),
top3 hit@1 与 regret 反而更优。**权衡的存在性有 τ 证据,权衡的服务代价
只能在 D0 上量化。**

**E13 —— 表示侧干预的完整负结果账(四类 vs 两类)。**

| 干预类别 | 实例 | 结果 |
|---|---|---|
| 模型特征 | F1–F4(砍 desc/fam、+e_task、name 投影) | 全 REJECT,z_m PR 不动 |
| 查询压缩 | Z1(4618→128 可学习投影) | REJECT,z_d PR 反降 |
| 查询几何约束 | Z2(异任务斥力) | REJECT,PR +0.1 |
| 叠加于候选 | Z1L/ZL | **显著减益(ΔCI [−0.119, −0.018])** |
| **目标/负样本分布** | G1(任务不兼容池) | **0.010→0.054(5.4×)** |
| **目标/负样本分布** | L1/L1L3(logQ 全湖) | **0.046→0.128(2.8×,CI 排除 0)** |

六组实验、两年惯性的"改特征/改结构"路线,与两次"改监督分布"的对照:
**有效杠杆只有一类。** Z 叠加减益的机理也已定位:lake 目标正是在利用高维
自由度分离 gold 与 hub,压缩恰好没收它——**目标与表示容量在小图上是竞争关系**。

### 0.2 v4 决策原则(四条)

1. **唯一主战场 = 监督分布**,且现在明确它有两侧:负样本侧(logQ 全湖,已做)
   与**正样本侧(尚未做,W2/L4)**——top-frac 正样本集合本身也是 hub 偏置的,
   逆倾向加权是 logQ 的对偶,这是 v4 仅存的新理论杠杆。
2. **表示侧冻结。** 不再手工设计 z 空间(特征、投影、几何约束一律不动)。
   重访的门槛条款:必须先给出"某几何性质是目标函数**不可能**自发达成"的
   诊断证明,才允许开表示行(防止回到 E13 左列的惯性)。
3. **权衡轴是 v4 的中心悬案。** hit@1/τ 的代价是否真实、多大、是否可用
   λ_global 调和——全部只在 D0(613 gold 节点,功效 ~10×)上裁决;
   2K 上不再为此加行。
4. **切分纪律升级:root-aware splits。** D0 图上 per-language 亲缘节点
   (tatoeba 112 个、massive 102 个)若跨切分即是近重复泄漏——同根节点
   必须整根同侧。这是 v2"按根分层报告"的硬化:从报表纪律升级为切分纪律。

### 0.3 工作包

**W1 —— D0 落地(critical path,其余一切服务于它)**

| 步 | 内容 | 关键决策/风险 |
|---|---|---|
| C1 湖定稿 | 以 **strict 口径 9,491(model-index ∨ lineage)** 为主湖;family-only 6K 不入图(用户已定调可弃);2,000 数据集节点 = gold 613 全入 + 任务分层补充 | 口径切换只是 `d0_model_intake.csv` 一列 flag |
| C2 特征重建 | xd0(gpt-neo 探针,GPU)/ xm0 / e_task vocab(d1 工具 `--graph` 重跑)/ task_type 修复(enrichment 工具对 D0 数据集重跑) | 修复覆盖率决定 L3 增益上限 |
| D1 建图 | trained_on 直接从 `d0_observations.parquet` 生成(源头去重);双契约(Stage-1 sanity + Stage-2 mappedID/vocab/逆向边/四边型) | 42% 重复边教训已在 parquet 层解决 |
| D2 切分 | **root-aware fixed splits**(新实现):同根数据集节点整根同侧,split_seed 三套照旧 | 不做即前功尽弃(近重复泄漏) |
| E1 训练侧 | pyg-lib/torch-sparse 安装;**lake 全图前向的规模化**:15K 图上每步全图 forward 不可行 → z 缓存隔 k 步刷新(k≈4,滞后负样本是采样 softmax 的标准近似)或负样本子图前向 | 这是 L1L3 上 10K 图的主要工程风险 |
| E2 基线 | 三列报告(B0 / G2 / **L1L3**)× 按根宏平均 × conditioning gain × overlap × z_d PR | 晋升阈值在此定标(终局条款照旧) |
| F 重索引 | 胜出行 export → build → query,fidelity ≥0.99 | 管线零改动 |

**W2 —— L 轨精修(2K 上便宜,先行;胜者随 W1 上 D0 复验)**

| 行 | 内容 | 理由 |
|---|---|---|
| L1b | n_neg 64→256 | 采样 softmax 的分母估计方差随样本数线性降;最便宜的一行 |
| **L4** | **IPW 正样本**:正样本项按 1/(deg(m)+1)^β 加权(β≈0.5 起步) | 负样本侧已 logQ 校正,**正样本侧的 hub 偏置原封未动**——top-frac 集合里 hub 模型占比高,它们作为正样本仍在把"平均查询方向"拉向自己。对偶修正闭合最后一个已知偏置源 |
| L2b | 带 alibi 的硬负样本:挖掘集 ∩(高度数 P90 ∨ 任务不兼容)\ P_d | 修 L2 自我拆台:只把"有流行度嫌疑或任务不符"的现胜者作负样本,真 gold(低度数、任务相符)不再被误伤 |
| L5(可选) | λ_global ∈ {0.5, 2.0} 单变量 | 仅当 D0 证实 hit@1 代价真实时启用(权衡旋钮,非涨点行) |

**W2 执行状态(2026-07-19,详见 `v4/W2_AUDIT.md`):** 三行已跑完。
**L1b 方向胜出并更新携带配置 → L1L3b(= L1L3 + n_neg 256)**:hit@1 完全恢复
0.401(=G2)、top3/regret 历史最优、gold@10 名义 0.155——E12 权衡轴疑似是
n_neg=64 的估计噪声而非本质,由 D0 终裁。**L4 证伪**(CI 显著负;正样本是
观测事实非采样分布,IPW 语义不成立,入 §0.4 字典);L2b 修复成功但零增益,
挖掘管线退役。L5(λ 调和)因 L1b 已回收 hit@1 而大概率不再需要。

**W1 执行状态(2026-07-19,详见 `v4/W1_D0_AUDIT.md`):** D0 落地完成——
strict 湖 9,491 / 1,463 节点(gold 613 全入)/ 54,795 边 / task Other 3.1%
(L3 建图时原生做对);root-aware splits 实装(泄漏断言内置);LightLinkLoader
在 11K 图实测可行(z 缓存留待 47K);重尾实测 max_deg 638、α=0.75 未失控。
**8-seed 终裁:L1L3b 依 root_macro 正式口径晋升 D0 默认候选**——五指标全 ≥ G2
(root gold@10 0.053→0.100 +89%,6/8 seeds;gold@1 3×;hit@1 打平),flat
hit@1 −0.085 为已定量的构成效应(大根族组内排序,不阻塞晋升)。B0 地板:
全湖 gold@10 = 0.000。E12 权衡轴终答:按根公平计权下权衡不存在。
z_d PR 阶梯 1.74→2.85→3.58 复现 E11 于 5 倍规模。

**W3 —— 卫生轨(一行)**:D2 P2 degree-cap 在 L1L3b 上跑一次(消息图卫生,
预期中性,验证即可)。

**W3 + Stage-3 执行状态(2026-07-19,详见 `v4/W3_STAGE3_AUDIT.md`):**
Stage-3 重索引完成——L1L3b@D0 冻结(功能门 replay Δ=0.0000;位级 sha 门因
CUDA scatter 非确定在 54.8K 边不可达,诚实降级并记录)、export/build **零改动**、
fidelity r@1..100 全 1.000。**W3 P2 终判:8-seed 中性,不采用**(3-seed 曾现
强假信号,root gold@10 +46%,补种子后均值回归至 0.100 vs 0.097、4/8——E10
纪律防止一次误晋升);"预期中性"确认,L1L3b 维持唯一晋升配置。
**v4 排期至此全部执行完毕。**

**W4 —— 评测升级**:root-aware splits 实现 + 按根宏平均进 harness;
overlap / conditioning gain / z_d PR 升为常设诊断列;D0 基线分布出来后
定数值晋升阈值(v2 §10 终局条款不变)。

### 0.4 明确不做清单(负结果字典,防惯性回流)

| 不做 | 证据 |
|---|---|
| F1–F4 任何特征变体(含 F1b/F3b 重访) | E7;E13;重访门槛条款(§0.2-2) |
| Z1/Z2 原参数及其任何"调参再试" | E11/E13,叠加显著减益 |
| L2 原版(无 alibi 的 occupancy 挖掘) | 自我拆台已定位(L 轨审计 §6.4) |
| 后置 CSLS/hubness 重排 | v2 E5,多样性≠相关性 |
| occupancy 对抗式 loss 减分 | v2 §6 反对理由仍成立(与 logQ 的原理性校正不同) |
| 2K 上任何 <0.05 差异的调参行 | E10,噪声以内 |

### 0.5 排期

```
第 1 步   W2:L1b / L4 / L2b 三行在 2K(对照 L1L3,~3×1h)——2026-07 内
第 2 步   W1 C1–D2:D0 湖定稿、特征重建、建图、双契约、root-aware splits
第 3 步   W1 E1–E2:pyg-lib + lake 规模化 + 三列基线(B0/G2/L1L3+W2胜者)
          → 在此定 D0 晋升阈值,并终裁 hit@1 权衡(E12)
第 4 步   胜出行终裁 + W3 卫生行 + Stage-3 重索引(G-D0 验收闭环)
之后      仅当 §0.2-2 门槛条款被满足才考虑任何表示行
```

### 0.6 风险登记(v4 新增项)

1. **lake 全图前向在 15K 图上的成本**(W1-E1):z 缓存隔 k 步刷新是标准
   近似,但滞后 z 与当前参数的错位需监控(loss 曲线毛刺);
2. **D0 度数分布更重尾**:2K 上 max_deg=13,D0 标签对 56K/模型 4.8K,hub
   度数将达数百——α=0.75 的温度化在重尾下需复核(记录采样熵,必要时 α 单变量);
3. **task 修复覆盖率在 D0 上未知**:enrichment 工具对 D0 数据集的三层证据
   命中率待实测,L3 增益可能随覆盖率缩放;
4. **root-aware splits 会减少可评测试根数**(整根同侧),按根功效需重估
   ——但这是把泄漏换成诚实,方向不容妥协。

---

## v3 修订(2026-07-18,已执行完毕,由 v4 收束)—— 如果改 embedding 特征没用,那什么才有用

**触发:** D1 §5.4 消融 F1–F4 全部 REJECT(完整记录
`D1_FEATURE_REWORK_AUDIT.md`),迫使重新回答根本问题:
**clean gold@10 ≈ 0.05、gold 中位排名 ~140,到底卡在哪。**

### 0.1 新证据(E7–E10,全部可复现)

**E7 —— F-phase:模型侧特征不是约束。** 砍 desc(F1)gold@10 微升
0.046→0.053 但 hit@1 掉 0.056;+e_task 9 值粗表(F3)在特征稀薄时把 gold@10
压到 **0.000**(塌缩加速器);z_m participation ratio 在全部五行里恒为 3±0.5
——**没有任何特征改动改变了嵌入空间的有效维度**。

**E8 —— 28 行历史全景:loss 是唯一动过 gold@10 的杠杆。** 扫描
`artifacts/ablation/top1/**/[row].json` 全部 28 个历史配置(架构深度、边权、
分头、edge-aware、dm-contrast、lr、early-stop、特征五行):

| 干预类别 | gold@10 范围 |
|---|---|
| 架构/边/超参(B*、R_*、P6、P7,17 行) | 0.010 – 0.047 |
| 特征(F1–F4) | 0.000 – 0.054(且 hit1 全回退) |
| **G1 全局负样本(loss 侧)** | **0.010 → 0.054(5.4×,同底对照 R_mg02)** |

复现:

```python
import json, os, glob
for p in sorted(glob.glob(r'ModelLakeFishing\stage2TrainGraphSAGE\artifacts\ablation\top1\**\*.json', recursive=True)):
    try: d = json.load(open(p, encoding='utf-8'))
    except Exception: continue
    a = d.get('aggregate_over_splits')
    if a and 'full2k_gold@10' in a:
        print(p, [round(x,3) for x in a['full2k_gold@10']])
```

**E9 —— 监督盲区:训练目标从未见过 serving 的失败对。** 这是 v3 的核心诊断。
每数据集带标签候选中位数 17;而 serving 时占满 top-10 槽位的 129 个 hub 模型,
对绝大多数查询数据集**没有标签**(E2:massive_intent 的 top-10 有标签数 0/10)。
RankNet/对比损失只在带标签候选**内部**构造序对——
**"gold 应该排在那 129 个 hub 前面"这个 serving 时真正要做对的比较,
训练时一对样本都不存在。** G1 的任务不兼容负样本是第一次部分修补,
但只覆盖 30/64 个受监督数据集(E3 的 task_type=Other 缺口),
恰好解释了它为什么只把 gold@10 抬到 0.054 就停了。

**E10 —— n=61 的统计功效不足。** gold@10 的跨 split std ≈ 0.035,
61 个 gold 数据集下 <0.05 的差异根本不可分辨——过去 28 行里大量
"0.018 vs 0.028"式比较在噪声以内。D0 合并湖已有 **613 个 gold 节点 /
211 个 gold 根**(`D0_LAKE_EXPANSION_LOG.md`),把评测功效放大 ~10×,
这不只是扩湖,是**让任何 gold@10 结论第一次变得可统计**。

### 0.2 重新诊断:三个真正的约束(按优先级)

```
score(d, m) = z_d · z_m  ≈  [全局流行度方向分量] + [条件化分量]
```

1. **目标函数盲区(E9)** —— 条件化分量没有学习信号:压过 gold 的对手
   (unlabeled hubs)从不作为负样本出现。这是 loss 问题,不是特征问题。
2. **z_d 塌缩(既有证据)** —— z_d 内在维度 ≈ 2、57.6% 候选重叠、静态榜
   解释 replay 大半:查询向量表达不出"我是哪个任务的哪种数据"。
   diverse-zoo 已证明瓶颈在数据集嵌入(gpt-neo 探针互相似)而非数据集数量。
3. **评测功效(E10)** —— 2K/61 上任何 <0.05 的改进都不可信;
   结论必须搬到 D0 图上做。

**x_m 特征重构从"主战场之一"降级为存档**:F 行负结果已记录,z_m PR 不变
说明模型侧输入不是有效维度的约束;10K 图 + z_d 修复后可重访(F1b/F3b 候选行
保留在审计文档里)。

### 0.3 v3 主攻方向(替代 v2 §3 的权重表)

| 轨 | 内容 | 为什么是它 | 代价 |
|---|---|---|---|
| **L —— 训练目标重构(主战场)** | **L1**:全湖 logQ 校正采样 softmax——gold 锚定的 InfoNCE,负样本从**整个模型湖**按标签度数分布采样,logQ 修正(`s − log q(m)`)直接中和流行度偏置(双塔检索的标准解法,也是 label-degree bias 的原理性对冲);**L2**:serving 硬负样本——用 export 的 occupancy sidecar 把"当前压过 gold 的 top-K 模型"作为负样本,**只做一轮**(P3 同款防循环纪律);**L3**:数据集侧 task_type=Other 修复(v1/v2 既有项,升级为 L 轨依赖:让任务不兼容池覆盖全部数据集而非 30/64) | E8:loss 是唯一动过 gold@10 的杠杆;E9:L1+L2 恰好把缺失的失败对喂给训练;G1 只做了一半(30/64 覆盖 + 均匀采样无度数校正) | 纯 loss 改动,2K 图上即可验证方向,便宜 |
| **Z —— z_d 条件化重建(第二战场)** | **Z1**:降 gpt-neo 探针 768 维的容量霸权(可学习投影 →128,xd0 离散表 task/nclass/arity 占比从 3% 提到 ~20%);**Z2**:数据集-数据集 push-apart——不同任务的数据集对作为 z_d 侧负样本(现在 similar_to 边只会拉近、没有任何目标推开不同任务的 z_d,内在维度 2 是这个单向力的直接后果) | z_d 塌缩是候选重叠 57.6% 与静态榜 0.656 的直接成因;模型侧改完没用(E7),条件化的另一半在查询侧 | Z1 便宜;Z2 需要 xd0 task 元数据(与 L3 同一依赖) |
| **D0 训练落地(试金石,不变)** | 10K 图建图+训练+基线三列报告;gold 报表**按根数据集分层**(tatoeba 一族不得淹没宏平均) | E10:没有它,一切 gold@10 结论都在噪声里 | 已完成 Phase A/B(入湖 15,210 / gold 613 节点),剩建图与训练 |
| D2 剪枝(降级为卫生轨) | 只保留 P2 degree-cap 一行,在 L/Z 胜出配置上跑一次验证消息图卫生 | 剪枝防 hub 聚合,但 E9 说明主要病灶在监督分布而非消息图;不再作为独立战场 | 一行 |
| 后置精排 | 维持 v2 判断:serving 可保留层,非研究方向 | — | — |

**执行状态(2026-07-18/19,详见 `v3/` 审计):** L 轨已执行——**L1 logQ 全湖
采样首破 28 行历史天花板(gold@10 0.046→0.110),L1L3 = 0.128(2.8×,CI 排除 0)
冻结为 v3 候选**;hit@1 名义回退在 n=61 下统计不可分辨,终裁移交 D0。
L2 挖掘存在"gold 自我拆台"缺陷,弃用待重设计。**Z 轨已执行并证伪(2K)**:
Z1 投影使 z_d PR 2.4→1.8(方向反了),Z 叠加 L1L3 使 gold@10 显著回落
(Δ CI [−0.119, −0.018])——目标函数与表示容量在 2K 上是竞争关系;
Z 假设的重访排在 D0 之后。

**一句话回答"到底该如何真正增加 gold@10":**
把 serving 时真正的比较对(gold vs 压过它的 unlabeled hubs)变成训练时的
负样本对(L1/L2),同时让 z_d 有能力把不同任务的查询分开(Z1/Z2),
并且在评测功效足够的 D0 图上下结论(613 gold 节点)——
**特征、架构、剪枝都不是当前的绑定约束,监督分布和查询侧几何才是。**

### 0.4 验收与纪律(继承不变)

- clean 固定 split 五指标仍是唯一晋升标准;三列基线 + conditioning gain@K
  照旧;one change per row 照旧;Stage-3 管线零改动照旧。
- L1 的第一个 2K 验证行:同 G2 底座只换全局负样本为 logQ 校正全湖采样,
  预期方向 = gold@10 越过 0.054 天花板且 hit@1 回退 ≤0.02;
  若 2K 上方向正确,一切后续行搬到 D0 图。
- 阈值纪律:D0 基线分布出来之前,不设具体数值门槛(v2 §10 终局条款不变)。

### 0.5 v3 排期

```
第 1 步   L3/Z 依赖:数据集侧 task_type=Other 修复(既有方案,直接执行)
          L1 实现 + 2K 方向验证(G2 底座,一行)
第 2 步   L2(occupancy 硬负样本,一轮)+ Z1(探针降维)2K 行
第 3 步   D0 建图(trained_on 从 d0_observations.parquet 生成)+ 双契约
          + pyg-lib 安装 + B0/G2/L1 基线三列报告(按根分层)
第 4 步   胜出 L×Z 组合行在 D0 图上复跑;Z2 视 Z1 结果决定
之后      D2 P2 卫生行;F1b/F3b 特征重访(仅当 L/Z 后仍有 headroom)
```

---
**触发:** dataset-6(`amazon_massive_intent`)审计 —— Phase-3 演示检索出语义完全错配的
top-10(全是 sentiment/SST2 hub 模型),而真正的 gold 模型排在 111–186 名。完整审计见
`weeks/week7_HNSW/HNSW.md` §"Phase 3 addendum"。
**v2 相对 v1 的根本变化(依据 `my_ideas.txt` 五条改进思路):**
主叙事从"扩 K + 精排兜底、几何修复并行"改为 **"直接动 Stage-1 的特征构建与
Stage-2 的图训练逻辑,同时把湖扩大到 10K models × 2K datasets"**。
后置精排(v1 Track C)降级为最终系统可保留的 serving 层,不再承担研究目标;
K=200 交接只是工程兜底。**动刀位置在 HNSW 之前,不在 HNSW 之后。**

---

## 1. 证据基础(每条标注复现方式)

当前服务栈:G2 嵌入 → HNSW(fidelity 1.000)→ cosine top-K。

**复现前置:** 所有指令在仓库根目录 `d:\research\model_lake\codes`、项目 venv
(`ModelLakeFishing\.venv`,PATH 已含)下运行。E2–E6 基于 Phase-1 导出
`ModelLakeFishing/stage3HNSW/artifacts/exports/hf1000d_G2/`(其 `manifest.json`
绑定了图与 checkpoint 的 sha256,输入身份可验)。以下片段共享的公共开头:

```python
# 公共开头(E2/E3/E4/E5 复用)
import numpy as np, torch, pandas as pd
exp = r'ModelLakeFishing\stage3HNSW\artifacts\exports\hf1000d_G2'
z_m = np.load(exp + r'\z_m.npy'); z_d = np.load(exp + r'\z_d.npy')
mids = pd.read_csv(exp + r'\model_ids.csv')['unique_model_id']
p = torch.load(r'ModelLakeFishing\stage1BuildTransferGraph\hgraph_hf1000d_2000m_xm0_xd0.pt',
               map_location='cpu', weights_only=False)
from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on
data = dedup_trained_on(p['data'])                      # 去重后的标签图(7,056 对)
ei = data[('model','trained_on','dataset')].edge_index.numpy()
ea = data[('model','trained_on','dataset')].edge_attr.numpy().ravel()
S = z_d @ z_m.T                                         # [362, 2000] 服务打分
```

### E1 —— clean 协议全湖数字(晋升口径)

| 事实 | 数字 |
|---|---|
| 全湖 gold 存活率 @10 | ≈ 0.05(G2 三 split 均值 0.046 ± 0.035) |
| gold 中位排名(split 0/1/2) | 141 / 127 / 196 |

来源:G-phase 五指标评测记录(固定 split、无泄漏),非 replay。复现:

```python
import json
d = json.load(open(r'ModelLakeFishing\stage2TrainGraphSAGE\artifacts\ablation\top1\G\G2.json'))
print(d['aggregate_over_splits']['full2k_gold@10'])     # [0.0457, 0.0352] = [mean, std]
print({s: v['aggregate']['median_gold_rank'] for s, v in d['splits'].items()})
                                                        # {'0': 141, '1': 127, '2': 196}
```

### E2 —— dataset-6(amazon_massive_intent)审计

| 事实 | 数字 |
|---|---|
| 该数据集带标签模型数 | 665 |
| gold(NV-Embed-v2 / SFR / stella,acc ≈ 1.0)的 G2 检索排名 | 111–186 |
| 检索 top-10 中有该数据集标签的 | **0/10** |
| hub 集中度 | 第一名 hub(poem-sentiment)在 **305/362(84%)** 数据集的 top-10;全部 3,620 个 top-10 槽位仅由 **129** 个模型占满 |

复现(接公共开头):

```python
D = 6; sel = ei[1] == D
models, accs = ei[0][sel], ea[sel]
rank_of = np.empty(2000, int); rank_of[np.argsort(-S[D])] = np.arange(1, 2001)
for i in np.argsort(-accs)[:8]:
    print(f'acc={accs[i]:.3f} rank={rank_of[models[i]]:4d} {mids.iloc[int(models[i])]}')
top10 = np.argsort(-S[D])[:10]
print('top-10 有标签?', [int(m) in set(models.tolist()) for m in top10])   # 全 False
all_top10 = np.argsort(-S, axis=1)[:, :10]
occ = {int(m): int((all_top10 == int(m)).any(axis=1).sum()) for m in np.unique(all_top10)}
print('distinct in top-10 slots:', len(occ), '| max occupancy:', max(occ.values()), '/362')
```

### E3 —— 元数据缺口:massive_intent 的 task_type = `Other`

后果:G2 的任务不兼容全局负样本对它从未生效(负样本池审计:仅 30/64 个受监督
数据集有非空不兼容池,见 G1.json 的 `[global negs]` 训练日志),任何任务硬过滤
在此同样失明。复现(接公共开头):

```python
ttv = {v: k for k, v in p['xd0_meta']['task_type_vocab'].items()}
print(ttv[int(data['dataset'].task_type_id[6])])        # -> Other
```

### E4 —— 三列基线、候选重叠、gold 多样性(replay 口径)

61 个可评 gold 数据集(≥3 标签、非常数 acc),池 = 2,000:

| K | 随机 (K/2000) | 静态全局榜 | 逐数据集(G2) |
|---|---|---|---|
| 10 | 0.005 | 0.016 | 0.230 |
| 200 | 0.100 | 0.295 | 0.656 |
| 500 | 0.250 | **0.656** | 0.869 |

- 61 个查询两两 top-500 平均重叠 **57.6%**——一半以上候选与查询无关;
- 静态榜(对所有查询返回同一份榜单)gold@500 = 0.656——**popularity 解释了
  moderate-K recall 的大头**;G2 条件化增益真实(每档比静态榜高一倍上下)但被稀释;
- 51 个不同 gold / 61 数据集——gold 多样性不是问题,病在排序几何。

复现(接公共开头):

```python
golds = {}
for d_ in np.unique(ei[1]):
    s = ei[1] == d_
    if s.sum() >= 3 and np.std(ea[s]) > 0:
        golds[int(d_)] = int(ei[0][s][np.argmax(ea[s])])
static = np.argsort(-S.mean(axis=0))
static_rank = np.empty(2000, int); static_rank[static] = np.arange(1, 2001)
per = np.array([int((S[d_] > S[d_, g]).sum()) + 1 for d_, g in golds.items()])
sr = np.array([static_rank[g] for g in golds.values()])
for K in (10, 200, 500):
    print(K, 'G2', round(np.mean(per <= K), 3), '| static', round(np.mean(sr <= K), 3),
          '| random', K / 2000)
import itertools
tops = {d_: set(np.argsort(-S[d_])[:500].tolist()) for d_ in golds}
print('overlap:', round(np.mean([len(tops[a] & tops[b]) / 500
      for a, b in itertools.combinations(golds, 2)]), 3))     # 0.576
print('distinct golds:', len(set(golds.values())), '/', len(golds))   # 51/61
```

### E5 —— CSLS 教训:打散 ≠ 提相关

CSLS 式 hub 修正(`s' = 2·s − r_m`,r_m = 模型对其 10 个最近查询的均值相似度)
把 top-10 槽位内 distinct 模型从 129 提到 **186**、observed hit@1 从 0.426 提到
0.443,但 **gold@10 反降(0.230→0.213)**——"多样性上升"≠"相关性上升"。
本方案所有去 hub 手段(§6)的验收都固化此教训。复现(接 E4 的 `golds`):

```python
r_m = np.sort(S, axis=0)[-10:, :].mean(axis=0)          # [2000]
S2 = 2 * S - r_m[None, :]
per2 = np.array([int((S2[d_] > S2[d_, g]).sum()) + 1 for d_, g in golds.items()])
print('CSLS gold@10:', round(np.mean(per2 <= 10), 3))   # 0.213 (raw 0.230)
t2 = np.argsort(-S2, axis=1)[:, :10]
print('CSLS distinct in top-10 slots:', len(np.unique(t2)))   # 186 (raw 129)
```

### E6 —— 模型特征维度拆解(§5 特征重构的直接依据)

实测 `x_m^(0) = [e_name 64 ‖ e_desc 384 ‖ e_size 16 ‖ e_fam 16]` = 480 维:
**desc 一项占 80%**(HF model card 大量为模板文本,MiniLM 向量彼此高度相似,
高维弱信号),**可学习的 size+fam 仅占 32/480 ≈ 7%**——"有效信号约 25%"的
直觉比实际还保守。同时:数据集侧早有 task_type 表(10 类,16 维),
**模型侧完全没有任务特征**——e_task 补的正是这个结构性缺口,且可完全复用
family vocab 基建(Other=0 兜底、vocab 绑 checkpoint、归纳性不破)。复现:

```python
import torch
ck = torch.load(r'ModelLakeFishing\stage2TrainGraphSAGE\artifacts\ablation\top1\ckpt\G2_s0_i0.pt',
                map_location='cpu', weights_only=False)
print({k: v for k, v in ck['arch'].items() if k != 'metadata'})
# frozen_dim=448, size_dim=16, family_dim=16, num_families=185,
# num_task_types=10(数据集侧), task_dim=16(数据集侧)—— 模型侧无 task
from ModelLakeFishing.stage2TrainGraphSAGE import learnable as L
print(L.DEFAULT_ENAME_TOKEN_DIM, L.DEFAULT_EDESC_ENCODER)
# 64, all-MiniLM-L6-v2  =>  e_desc = 448 - 64 = 384 维(占 384/480 = 80%)
p = torch.load(r'ModelLakeFishing\stage1BuildTransferGraph\hgraph_hf1000d_2000m_xm0_xd0.pt',
               map_location='cpu', weights_only=False)
print(p['data']['model'].x.shape)                       # [2000, 448] 冻结半
```

**口径警告(适用 E2/E4/E5):** replay 用的是全图服务嵌入(消息图含 clean
协议下的测试边),绝对值相对 clean 协议**偏乐观**(replay gold@10 = 0.230 vs
clean ≈ 0.05,E1),只用于同底相对比较与 K 的选型;一切晋升决策以 E1 的
clean 固定 split 五指标为准(§7.6/§8)。

## 2. 诊断:hub 点膨胀是根本问题(思路 §1)

**这一点 proposal 早有预料。** proposal 的 "Potential Issue § hub-dominated graphs"
和 CLAUDE.md 的三大风险之一("Over-smoothing near hubs")当初就是为此写的;
`L_contrast` 的同族重负样本设计也是为此。但已有对策只覆盖了
**同一 lineage 家族内衍生模型的塌缩**,没有覆盖本次审计暴露的另一个来源:

1. **训练分布偏差(标签度数偏差)。** 少数模型(MTEB 嵌入模型、热门微调模型)
   在很多数据集上有 `trained_on` 标签。RankNet/对比训练在这种分布上会把这些
   模型推向"平均查询方向"——它们对每个查询都赢。这不是模型架构问题,
   是**图里的边分布问题**,所以在图上动刀(§6)比在 loss 上贴补丁更对症。
2. **z_d 条件化不足。** z_d 内在维度 ≈ 2(diverse-zoo 诊断),不同数据集问的
   几乎是同一个问题,自然得到几乎同一批(hub)答案 —— 57.6% 的候选重叠和
   静态榜 0.656@500 都是它的直接表现。模型侧特征重构(§5)间接改善这一点:
   模型侧有了任务特征,dataset-model 的任务对齐才有可学的支点。
3. **特征信噪比低。** 模型输入 480 维里 93% 是冻结的 name/desc 文本嵌入,
   其中 desc 一项占 80%(见 §5 的精确拆解)——弱信号维度淹没强信号维度,
   GNN 只能更多依赖图结构,而图结构又被 hub 边分布污染,形成恶性循环。

## 3. 方向总览与权重(思路 §0/§2)

| 方向 | 内容 | 权重 |
|---|---|---|
| **D0 扩湖** | 10,000 models × 2,000 datasets | **≈ 其余全部之和** |
| **D1 模型特征重构** | 砍 desc/fam,加 e_task,name 结构性降权 | 主战场之一 |
| **D2 图手术去 hub** | 剪边(per-model top-k / hub 检测剪枝),不做 loss 减分 | 主战场之一 |
| 度量仪表 | 三列基线、conditioning gain、overlap、occupancy、clean 协议 | 不变,全部方向的验尺 |
| Stage-3 管线 | export → build → query,checkpoint 不可知 | 零改动,每个胜出配方重索引即可 |
| 后置精排 | 原 v1 Track C | **降级**:serving 可保留层,非研究方向,推迟到 D1/D2 出结果后再评估 |

**干预点声明(思路 §2):** 不在 HNSW 之后动刀。HNSW fidelity 已是 1.000,
它忠实复现的是一个错误的排序;精排只能重排候选集内部,救不回没被召回的
专门化模型,也治不了候选集跨查询高度重复。**候选质量由 Stage-1 特征 +
Stage-2 训练决定,这就是本方案的全部动刀位置。**

## 4. D0 —— 扩湖:10,000 models × 2,000 datasets(思路 §0)

### 4.1 为什么它与其余全部等重

1. **监督稀疏是 hub 的成因之一。** 现在的 2K 湖只有 61 个可评 gold 数据集、
   每数据集带标签候选中位数 17。标签集中在少数模型上,hub 几何是这种
   分布的必然产物;湖扩大 5 倍、数据集扩大 5.5 倍,标签度数分布本身就会摊薄。
2. **2K 上的结论可能不外推。** 静态榜能解释 replay recall 的大半,部分原因是
   任务面太窄(362 个数据集里大量同任务近亲)。D1/D2 的任何几何改进如果只在
   2K 上验证,拿到 10K 上可能失效 —— 反过来,10K 湖本身就是 D1/D2 结论的
   试金石。
3. **proposal 的目标是 47K/1M。** 10K 是从 2K 通往 47K(ModelLens 基准)的
   必经中间规模,现在做正好承接 Stage-3 已经就绪的 O(log N) 服务层。

### 4.2 可行性:漏斗基建已在

effective-dataset v2 的 Phase 0–1 已经**离线**盘点过 `hf_cache/models` 的
model-index:标签组 ≥3 的有 **1,145 组**、≥10 的 696 组、≥20 的 629 组;
非 bitext 强覆盖 397 模型 / 66 数据集。也就是说 harvest、解析、
provenance 对账的管线都是现成的,扩湖主要是:

1. **放宽入湖标准 + 补采**:有 model-index 评测记录、或家族可解析、或有
   lineage 线索的模型均可入湖;目标 ≥10,000 models、≥1,500–2,000 datasets。
2. **标签构建沿用 v2 的 provenance-reconciled 管线** —— 必须继承
   `attributes.py` 连接导致 42% 重复边泄漏的教训(12,205→7,056 的对账),
   新图从源头就去重。
3. **特征离线构建**:xm0(含 §5 的新特征)/ xd0 全部离线;e_task 的 vocab
   在这一步一并建好(§5.3)。
4. **训练侧准备**:当前环境 pyg-lib/torch-sparse 未装,一直在用
   LightLinkLoader 全邻居回退 —— 2K 图无所谓,10K 图必须装上真正的
   `LinkNeighborLoader` 后端(这是 CLAUDE.md 早已预告的 47K 前提)。

### 4.3 验收(G-D0)

- 新图通过 Stage-1 sanity + Stage-2 接口契约(mappedID 行序、vocab 绑定、
  逆向边、四种边型);
- 可评 gold 数据集数从 61 提升到 **≥200**(≥3 标签、非常数 acc);
- B0 与 G2 配置在新图上重跑作为基线,**三列基线 + conditioning gain 全套报告**;
- Stage-3 重导出、重建索引,fidelity gate 照旧(预期 1.000)。

## 5. D1 —— 模型特征重构(思路 §3)

> **[v3 状态:已执行并存档,全行 REJECT。]** 实现与消融记录见
> `D1_FEATURE_REWORK_AUDIT.md`;结论(desc 非纯噪声、e_task 粗表塌缩、
> z_m PR 不变)已并入 §0.1 E7。基建(e_task vocab/encoder 变体/契约)保留可复用;
> F1b/F3b 重访仅在 v3 L/Z 轨完成后考虑。

### 5.1 现状的精确拆解(证据与复现见 §1 E6)

当前模型节点输入 `x_m^(0) = [e_name ‖ e_desc ‖ e_size ‖ e_fam]`:

| 分量 | 维度 | 占比 | 性质 | 判断 |
|---|---|---|---|---|
| e_name | 64 | 13% | 冻结,hashed(10K 桶,seed 固定) | 信号在但表达粗糙(见 5.2 的警告) |
| e_desc | 384 | **80%** | 冻结,MiniLM(all-MiniLM-L6-v2)编码 model card | **疑似噪声主体**:HF model card 大量是模板文本("This model is a fine-tuned version of …"),MiniLM 向量彼此高度相似,高维弱信号稀释一切 |
| e_size | 16 | 3.3% | 可学习(15 桶) | 有效 |
| e_fam | 16 | 3.3% | 可学习(185 族) | 弱:185 族严重长尾,rare rows 学不到东西(CLAUDE.md 早有预警的 `FAMILY_MIN_COUNT` 问题);且家族信息本就由 lineage 边承载,特征层是冗余通道 |

即:**可学习的强信号只占 32/480 ≈ 7% 的输入维度**,比"25% 有效信号"的
直觉还要严峻。另外一个结构性缺口:**数据集侧有 task_type 表(10 类,16 维),
模型侧完全没有任务特征** —— dataset-model 的任务对齐在特征层没有支点,
全靠图结构隐式传递。

### 5.2 动刀前必须承认的一个反例(诚实警告)

**e_name 是当前最强的任务信号载体之一。** 本次审计检索出的模型名里直接写着
`sst2`、`qqp`、`rotten_tomatoes`——微调模型的命名习惯把训练数据集写进了名字。
所以思路 §3 的方向(保留 name 但把它的占比压到尽可能小)是对的,
但"砍到只剩 name+size"里 name 的处理方式必须谨慎:**降权不等于降维before验证**,
先跑消融(5.4)确认各分量的真实贡献,再定 name 的最终形态。

### 5.3 新特征方案

目标形态:

```
x_m^(0)′ = [ e_name↓ ‖ e_size ‖ e_task ]
             64→16     16        16(新增,可学习)
```

1. **砍 e_desc(384 维)。** 模板噪声主体,一刀去掉 80% 的弱信号维度。
   预案:若消融显示 desc 对冷启动模型(无边、名字无信息)有残值,
   退一步保留但降到 32 维(线性投影/PCA),不回到 384。
2. **砍 e_fam(16 维)。** 家族信息由 lineage 边承载,特征层砍掉不等于丢失
   家族结构;185 行长尾表本身学不好。释放的容量给 e_task。
3. **新增 e_task(16 维,可学习)。** 从 HF json 构建模型侧任务 vocab:
   - 来源:`pipeline_tag` + model-index 里的 task 字段(`hf_cache/models`
     离线可得,v2 管线已在解析);
   - 机制**完全复用 family vocab 的基建**:离线建 `task_vocab.csv`,
     Other=0 兜底,`nn.Embedding` 表挂在 `ModelNodeEncoder` 里,
     vocab 与 checkpoint 绑定(身份凭证纪律不变);
   - 零样本新模型:pipeline_tag 缺失 → Other,与 size 的 unknown 桶、
     family 的 Other 完全同构 —— 归纳性不破;
   - 这同时就是 v1"元数据增强"在**模型侧**的落地(v1 只提了数据集侧的
     task_type=Other 修复;两侧都要修,dataset 侧的 `Other` 修复照旧进行)。
4. **e_name 结构性降权。** 一个必须写清楚的工程事实:encoder 是
   concat + Linear,**给 e_name 乘一个固定小系数会被第一层线性变换完全吸收,
   等于没降权**。真正的降权是**结构性的**:把 64 维 hashed name 经一个冻结的
   随机投影(或可学习但低秩的线性层)压到 16 维再拼接 —— 维度占比从 13%
   降到 16/64 = 25%(新总维 64),name 想主导也没有容量主导。
   可选变体:16 维投影 + 可学习标量门控(初始化小),让训练自己决定用多少。

新输入总维:64(对比原 480)。附带收益:第一层参数量骤降,10K 图训练更快。

### 5.4 消融矩阵(先在现 2K 图上验证假设,一次一行)

| 行 | 配置 | 验证什么 |
|---|---|---|
| F0 | 现状(G2 复刻) | 对照 |
| F1 | 去 e_desc | "desc 是噪声主体"假设 |
| F2 | F1 + 去 e_fam | "fam 作用不大"假设 |
| F3 | F2 + e_task | 任务特征的增量价值(**预期主要涨点来源**) |
| F4 | F3 + name 64→16 | name 降权的净效应(对照 5.2 的警告) |

每行报告:clean 五指标(晋升唯一标准)+ conditioning gain@K + pairwise
top-K overlap + hub occupancy + z_m/z_d participation ratio。
G2 的训练配置(RankNet、全局负样本等)全部冻结不动 —— **本节只动特征**,
遵守 one change per row。

### 5.5 验收(G-D1)

某个 F 行同时满足:clean 晋升规则通过(gold@10 全 split 上升或 paired CI>0,
observed hit@1 回退 ≤0.02,regret 恶化 ≤0.005)**且** conditioning gain@K 上升
**且** z_m participation ratio 不降。F3 是预期胜出行;若 F1/F2 单独就能过,
说明砍噪声本身就是涨点,特征越简越好。

## 6. D2 —— 图手术去 hub:剪边而非 loss 减分(思路 §4)

> **[v3 状态:降级为卫生轨。]** E9 表明主要病灶在监督分布(训练从未见过
> serving 失败对)而非消息图聚合;v3 只保留 P2 degree-cap 一行,在 L/Z 胜出
> 配置上验证一次。"剪边而非 loss 减分"的 v2 判断被 v3 部分推翻:
> **loss 侧(负样本分布)恰恰是历史上唯一有效的杠杆(E8)**——v2 反对的是
> occupancy 加权的对抗式减分,v3 采纳的是负采样分布的原理性校正,两者不同。

### 6.1 为什么剪边优于 loss 减分

v1 提过 occupancy 加权负采样(loss 侧对抗)。思路 §4 主张的剪枝是
**结构性预防**:hub 的成因是消息图里少数模型连着几百条 trained_on 边,
剪掉这些边,hub 在聚合阶段就不可能形成 —— 比训练期对抗更对症,
也更可解释(剪了哪些边一目了然)。CSLS 教训支持这个判断:
对抗式打散容易得到"多样性上升、相关性不动"。采纳:**剪边为主线,
loss 减分不做**(如剪枝失败再回头评估)。

### 6.2 铁律:只剪消息图,监督不动

`trained_on` 边有双重身份:**消息边**(GNN 聚合用)和**监督对**
(L_perf/RankNet 的标签)。剪枝**只作用于消息图**,监督对一条不丢 ——
否则等于扔训练数据。这与 `dedup_trained_on`、`similar_to` topk 手术是
同一模式,`graph_surgery.py` 的基建直接复用。同理,**lineage 边豁免**:
它是冷启动模型唯一的边(CLAUDE.md 给它低 dropout 特权,剪枝同样绕开)。

### 6.3 三个候选手术(one change per row)

| 行 | 手术 | 机制 |
|---|---|---|
| P1 | **per-model top-k trained_on 消息边** | 每个模型只保留 k 条"最匹配"的数据集连接,k ∈ {3, 5, 10}。匹配度打分:v1 版用 accuracy 边权;e_task 落地后升级为 accuracy × 任务匹配(思路 §4 强调任务)。NV-Embed 这类真全能模型的高分边保留,靠边数吃流行度的模型被截断 |
| P2 | **degree-cap(温和版)** | 只对度数超过阈值 τ 的模型动手,按边权截断到 τ;非 hub 模型完全不动。τ 从度数分布分位数定(如 P95) |
| P3 | **occupancy 反馈剪枝(谨慎)** | 用 export 的 occupancy sidecar 检测 hub(如出现在 >50% 数据集 top-10),对检出者砍低权边。注意:occupancy 是训练结果算出来的,存在"训练→测量→再训练"的循环依赖,**只做一轮**,且 occupancy 必须来自训练侧数据(不得用评测查询算,防泄漏) |

预期 P1(k=5,任务加权)是主行;P2 是保守备选;P3 是验证性实验。

### 6.4 验收(G-D2)—— CSLS 教训固化为三合一

**同时**满足,缺一拒收:

1. hub occupancy 显著下降(max occupancy、top-10 槽位集中度);
2. **conditioning gain@K 上升**(打散必须换来条件化,不是白打散);
3. clean gold@10 / gold@200 不降,observed 指标过晋升规则。

只满足 1 = CSLS 式假改善,拒收。

## 7. 度量仪表 —— v1 Track A 全套(所有方向的验尺,先于一切改动落地)

审计只是一次查询;在动任何特征/图/湖之前,把它变成常设仪器。
D0/D1/D2 的**每一行实验**都用本节仪表验收。工期 ~1–2 天。

### 7.1 两个测量工具

1. **`bench_params.py`(Stage-3 Phase 4 原计划)+ hub 分层 ANN recall
   (gate G-F)**:near-hub 与 away-hub 查询分开报告 ANN recall,
   用现有 `model_to_model_hnsw_recall` 的 near-hub 逻辑 + `top1_hubness.py`
   的 hub 列表。2K 上如果 near-hub recall 已经落后,就是 1M 的最早预警。
2. **gold-rank replay harness(新增,小工具,几分钟跑完)**:对每个带标签
   数据集输出——gold 模型在服务排序中的排名、gold@K 曲线
   (K ∈ {10, 50, 100, 200, 500})、hub occupancy 直方图(top-10 槽位内
   不同模型数、max occupancy)。每个 export 产出一份 JSON + 一张 MD 表。
   **这是 D0/D1/D2 必须推动的 KPI**——本文 §1 的全部数字今天是 ad hoc
   算的,验收标准之一就是 harness 能原样复现它们。
3. 两者接入 Phase-8 报告模板(STAGE3_REPORT / 后续每个 checkpoint 报告)。

### 7.2 三列基线强制并排

每个 K ∈ {10, 50, 100, 200, 500} 必须**同时**报告:

1. uniform random(K/N);
2. **static global ranking**(对所有查询返回同一份榜单的静态零假设);
3. per-dataset(被评 checkpoint)。

**禁止只与 uniform random 比较**——只有 static 基线能度量
"dataset-independent popularity 已经解释了多少结果"(§1 的表即为模板)。

### 7.3 conditioning gain@K(D0/D1/D2 共同的主 KPI)

定义:

```
conditioning gain@K            = gold@K(checkpoint) − gold@K(static)
normalized conditioning gain@K = (gold@K(ckpt) − gold@K(static)) / (1 − gold@K(static))
```

当前 G2 实测(replay,61 gold 数据集):

| K | gold@K(G2) | gold@K(static) | gain | normalized gain |
|---|---|---|---|---|
| 10 | 0.230 | 0.016 | 0.214 | 0.217 |
| 200 | 0.656 | 0.295 | 0.361 | 0.512 |
| 500 | 0.869 | 0.656 | 0.213 | 0.619 |

图训练的目标不只是提高 gold@K,而是提高**相对静态榜的 query-conditioned
增益**——gold@K 涨而 gain 不涨,说明只是 popularity 排得更准了。

### 7.4 候选集条件化指标(检查不同数据集是否真的拿到不同候选)

至少报告:

- mean pairwise top-K overlap(当前 top-500 = **0.576**);
- top-K Jaccard overlap 的**分布**(不只均值——双峰分布会被均值掩盖);
- occupancy 集中度:max occupancy、占满全部 top-10 槽位的 distinct 模型数;
- effective candidate diversity / occupancy entropy;
- 每个查询相对 static top-K 的**新增候选比例**(novel-candidate fraction)。

警告(CSLS 教训的度量化):**单独增加 distinct 模型数不够**——随机扰动
也能增加多样性,却未必增加相关性;上述指标必须与 §7.3 的 gain 联读。

### 7.5 两类 recall 严格分名

- **ANN recall**:HNSW 是否复现 exact-cosine top-K(系统层;当前 1.000);
- **semantic gold recall**:嵌入排序里含不含 gold(质量层;当前的病灶)。

报告中禁止把两者混称 "retrieval recall"——本次外部审计的误诊
(把嵌入几何问题当成检索器问题)正是这种混称的后果。

### 7.6 clean 协议补充 gold@K

现有 gold@200/500 全部来自乐观 replay。harness 需在 **clean 固定 split**
上补齐 gold@{50, 100, 200, 500}(`top1_eval` 的 `GOLD_KS` 目前只到 100,
扩到 500 即可)。所有表格必须显式标注两类数字:

- **clean promotion metrics**(晋升唯一依据);
- **optimistic replay diagnostics**(诊断与同底比较)。

replay 数字不得作为长期晋升门槛。

### 7.7 验收(G-A)

- `hf1000d_G2` 产出全套 JSON + MD(gold@K 三列并排、gain、overlap 分布、
  occupancy、G-F 分层);
- 本文 §1 的全部 ad hoc 数字由 harness 原样复现;
- Phase-8 报告模板引用这套输出;后续每个 checkpoint 自动带全套仪表数字。

## 8. 其他保留不变的部分

- **clean 固定 split 五指标是唯一晋升标准**(`top1_eval.py` + paired
  bootstrap),replay 只做诊断——§7 全套仪表不改变这一条。
- **Stage-3 管线零改动:** export → build → query 对 checkpoint 不可知,
  每个胜出配方 `export_embeddings.py → build_index.py` 重索引即可,
  fidelity ≥0.99 gate 照旧。这个闭环便宜正是当初把 Stage 3 做成
  checkpoint-agnostic 的意义。
- **K=200 交接与 occupancy sidecar:** 保留,定位为 serving 兜底与特征供给。
- **不做清单(继承 v1):** 不做 num_labels 硬过滤(会删掉本例 gold ——
  无头嵌入模型);task 元数据修好前不做任务硬过滤;不调 efS/M 对抗 hub;
  replay 数字不做晋升门槛。

## 9. 排期与依赖

```
第 1 周      §7 仪表落地(gold-rank replay harness + 三列基线 + gain)← 先于一切
             F1/F2 消融(验证 desc/fam 假设,现 2K 图,便宜)
             e_task vocab 离线构建(HF json → task_vocab.csv,两侧一起修)
             D0 启动:入湖标准定稿 + harvest 补采(后台跑)
第 2 周      F3/F4 行(e_task + name 降维)
             P1/P2 剪枝行(现 2K 图;P1 的任务加权版排在 e_task 之后)
第 3–4 周    D0 新图落地(Stage-1 构建 + 双 stage contract 验证)
             胜出的 F×P 配方在 10K 图上复跑 + 组合行(单因素胜出者叠加)
之后         10K 图上重评:hub 分布、conditioning gain、
             以及是否还需要精排层(若第一阶段召回修好,精排任务会很轻)
背景         dataset 侧 task_type=Other 修复照旧;v2 湖漏斗持续
```

依赖:F3 依赖 e_task vocab;P1 任务加权版依赖 e_task;P3 依赖 export 的
occupancy sidecar(已有);D0 与 2K 图上的 F/P 消融完全并行。
纪律:**one change per row,组合行只叠加单因素胜出者。**

## 10. Gates 汇总

| # | 主张 | 判据 | 协议 |
|---|---|---|---|
| G-A | 仪表落地 | hf1000d_G2 全套报告存在(gold@K 三列并排、conditioning gain、overlap 分布、occupancy、G-F 分层)且复现 §1 全部数字 | replay + Phase 4 + clean 补充 |
| G-D0 | 湖扩大且干净 | ≥10K models / ≥1.5K datasets;双 stage contract 通过;可评 gold 数据集 ≥200;B0/G2 基线三列报告落地 | Stage-1/2 契约 + 仪表 |
| G-D1 | 特征重构有效 | 某 F 行:clean 晋升规则 ∧ conditioning gain↑ ∧ z_m PR 不降 | clean 五指标 + 仪表 |
| G-D2 | 剪枝去 hub 有效 | 某 P 行:occupancy↓ ∧ conditioning gain↑ ∧ clean gold 不降(三合一,缺一拒收) | 同上 |
| G-SYS | 系统层不回退 | 每个晋升 checkpoint 重索引 fidelity ≥0.99(预期 1.000);row-order/manifest 契约全绿 | Stage-3 gates |
| 终局 | 10K 湖 + 新特征 + 剪枝的 G2′ 显著优于 2K G2 | clean gold@10 与 conditioning gain 的具体阈值,待 G-D0 基线分布出来后设定(不拍脑袋) | clean 五指标 |

## 11. 最终叙事(一段话版)

HNSW 系统层工作正常,它忠实复现当前嵌入的排序;G2 明显优于随机也明显优于
静态榜,说明图训练学到了真实的 dataset-conditioned 信号,但这部分信号被
全局流行度(hub 边分布)和 93% 的弱信号输入维度严重稀释。整改不在检索之后
贴补丁,而在检索之前动刀:**把湖扩到 10K×2K(摊薄标签度数偏差、扩大任务面)、
把模型特征砍到 [name↓ ‖ size ‖ task] 的高信噪比形态(task 来自 HF json,
复用 family vocab 基建)、在消息图上做 per-model top-k / degree-cap 剪枝
(监督不动、lineage 豁免)** —— 三者用同一套三列基线 + conditioning gain
仪表验收,每个胜出配方经不变的 Stage-3 管线重索引上线。去 hub 的目标始终是
相关性和条件化,不是让更多不同的模型进 top-10;两阶段结构保留,但研究贡献
集中在让第一阶段产出真正 query-specific、高召回的候选集。
