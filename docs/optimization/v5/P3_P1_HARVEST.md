# v5 P3+P1 可审计记录 —— 监督面扩容(D0.5)与查询侧内容特征采集

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 P3 / P1
**P0 授权:** O-task 0.080 < 现任 0.100 → 信息饥饿证实,P1/P3 为主攻
(`P0_DIAGNOSIS.md` §2.1);O-sibling 0.563 上界与 cold-gold 27.9% 是供给对象。
**性质:** 两条有界网络管线,全缓存断点续跑,仅采元数据/样本文本,不下权重/全量数据。

---

## 1. P3 —— 监督面扩容(D0.5)

### 1.1 原理与动作

v4 湖是 Phase-B 保守参数的产物:数据集保留门槛"邻域 ≥2 eval 模型"、
每数据集模型 cap 40。**跨根泛化的第一营养是更多的根**(P0:cold-gold 27.9%
结构性够不到,更多训练根是唯一补法)。P3 放宽两个旋钮,复用同一份**已全缓存**
的 seed 邻域扫描(12,153 次 list 调用),只有真正新增的 /api 拉取花网络:

| 旋钮 | v4 | P3 |
|---|---|---|
| 数据集保留门槛 min_keep | 2 | **1** |
| 每数据集模型 fetch cap | 40 | **100** |

### 1.2 seed 重导出(cache-only,即时)

```
min_keep=1: raw 6,362 → canonical 5,649 → keep 732(v4 为 530)
            keep_roots 730(v4 为 528)
```

保留数据集 +38%、保留根 +38%。产物 `d0_harvest_seed_p3.csv`(与 v4 的
`d0_harvest_seed.csv` 并存,不覆盖)。

### 1.3 代码改动

| 文件 | 改动 |
|---|---|
| `d0_online_harvest.py` | `phase_seed/datasets/models` 加参数(min_keep/seed_csv/fetch_cap);CLI `--min-keep`/`--fetch-cap`/`--seed-csv`;**默认逐位复现 v4 湖**(min_keep≠2 时才切到 `d0_harvest_seed_p3.csv`) |

### 1.4 补采结果(2026-07-19 完成)—— 增量真实但远低于预期,附诊断

datasets +145 JSON、models +908(planned 5,584,fail 6)。合并缓存后重跑
`d0_intake_audit.py`,D0.5 建图口径实测:

| 量 | v4 D0 | D0.5 | 增量 |
|---|---|---|---|
| 缓存模型 JSON | 15,318 | 16,226 | +908 |
| strict 模型(mi∨lineage) | 9,491 | **10,234** | +743(+8%) |
| 带标签数据集节点(bounded) | 1,463 | **1,514** | +51(+3.5%) |
| **训练根(建图口径)** | 421 | **461** | **+40(+9.5%)** |
| 监督边 | 54,795 | **55,140** | +345(+0.6%) |
| gold 可评根 | 211 | 212 | +1 |

**诚实诊断(方案 P3 的幅度假设被证伪):** 方案预期"根 421→800+、边 55K→100K+",
实测远未达到。根因不是保留门槛,是 **seed 扫描的广度**——那份已缓存的
候选池(14 类 × top-300 + 37 搜索 = 12,153 候选)本身就是绑定约束。
min_keep 2→1 只救回"恰好 1 个 eval 模型"的弱监督数据集(每个只带 ≤1 条边,
故边只 +345);cap 40→100 的 +908 模型多数附着到**已覆盖**的热门数据集
(加深而非拓宽)。**真正的根扩容需要更宽的 seed 重扫(新类目/搜索/lineage
爬取),是更大的网络作业,推迟。**

**据此的执行决策:** +40 根 / +345 边的幅度不值得单独一轮 D0.5 建图 + 16 次
重基线(几乎必然落在 v4 基线噪声内)。**P3 的增量并入 P1×P3 组合图一次性
验证**(方案 §0.5 步 3 的独立 D0.5 重基线取消,步 4 组合行内含 P3 扩容,
胜出则用消融归因);P1(内容特征)是 P0 指向的真实杠杆(O-sibling 0.563 缺口),
为组合图的主贡献者。

## 2. P1 —— 查询侧内容特征采集

### 2.1 原理

D0 图每个数据集节点只有一句元数据描述串(名字 + card task_categories + tags)。
root-aware 下测试根标签不可见,查询特征是唯一跨根通道——P0 证明该通道饥饿。
P1 供给缺失信号:**数据集的样本实际长什么样**。

### 2.2 管线(`p1_content_harvest.py`)

```
对每个数据集节点(mappedID 序):
  /splits 取首个(优先 train)config+split
  /rows offset=0 length=100 取样本行
  抽取文本字段(text/sentence/question/premise/... 命中,否则任意 string 字段)
  每行截断 400 字符,拼接存 d0_lake_cache/content/<node>.json
断点续跑(一节点一缓存文件);仅样本文本,不下全量。
```

**覆盖是部分的**(P0 探针 52%):无 served config 的节点存空样本,
特征化时回退元数据串——`has_content` 逐节点记录,P1 消融按有/无样本分层验收。

### 2.3 代码改动

| 文件 | 改动 |
|---|---|
| `p1_content_harvest.py` | **新增**:datasets-server rows 采样,文本字段抽取,断点续跑,产 `p1_content_index.csv` + report |

冒烟(前 20 节点):coverage 0.90、均 90 样本/节点(高知名度数据集在早
mappedID 聚集,全量将回归 ~52%)。

### 2.4 采集结果(2026-07-19 完成)—— 覆盖 21%,但结构性有利

全量 1,463 节点:**has_content 312(覆盖 21.3%)**,均 100 样本/命中节点。
远低于 P0 探针的 52%——诚实归因见下,但分层后画面截然不同:

| 分层 | 节点占比 | 内容覆盖 |
|---|---|---|
| **bitext/crosslingual 族**(tatoeba/ntrex/massive/…per-language) | **53.8%** | **6.7%** |
| 非 bitext(真实独立数据集) | 46.2% | **38.3%**(259/676) |
| gold 可评节点 | — | **26.6%**(164/616) |

**为何 21% 而非探针的 52%:** (1) P0 探针只查 `/splits` 可用性,P1 要端到端
抽出文本;(2) **图有一半是 bitext 兄弟节点**(per-language config),
datasets-server 对这些具体 language-config 几乎不供可抽取文本(6.7%),
它们把总覆盖拽了下来。

**为何这反而是有利结构(与用户问的"两条腿"吻合):** bitext 族是**兄弟众多**
的近重复家族——它们的信息通道是 O-sibling(0.563 上界正建立在有兄弟的根上),
**本来就不靠内容**;而真正需要内容的**非 bitext/孤根**恰是覆盖较高的那半
(38%),gold 节点也有 27% 拿到内容。**P1 的内容腿与 P3/sibling 的标签腿
覆盖互补,不是同一批节点。**

**据此的验收纪律(强化 §0.4-3):** P1 消融必须**在 has_content 分层上单独看
增量**——no-content 节点回退元数据串,是天然对照;任何 P1 效应若存在,
必须现于那 312 个(尤其 164 个 gold)有内容的节点。全局平均会被 79% 无内容
节点稀释,不作主判据。

## 3. 复现

```bash
cd ModelLakeFishing/stage1BuildTransferGraph
# P3(seed 重导出即时;datasets/models 仅新增部分花网络)
../.venv/Scripts/python.exe d0_online_harvest.py --phase seed     --min-keep 1
../.venv/Scripts/python.exe d0_online_harvest.py --phase datasets --min-keep 1
../.venv/Scripts/python.exe d0_online_harvest.py --phase models   --min-keep 1 --fetch-cap 100
# P1(1,463 节点,断点续跑)
../.venv/Scripts/python.exe p1_content_harvest.py --n-rows 100
```

## 4. 下一步(v5 排期第 3–4 步)

1. P3 完 → D0.5 建图(`d0_build_graph.py` 换 seed manifest + 观测重聚合)+
   现任 L1L3b 8-seed 重基线(新湖分母);
2. P1 完 → build_xd0 内容视图 + e_content 拼入 → L1L3b+content 行 8 seeds,
   按 has_content 分层验收;
3. P1×P3 组合行 8-seed 终裁,对齐 §0.6 承诺(root gold@10 0.22–0.29 /
   gold-gap@10 ≥0.45);胜者 Stage-3 重索引。