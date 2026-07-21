# D0 扩湖执行记录 —— Phase A 离线审计 + Phase B 在线补采

**日期:** 2026-07-18(Phase A 与 Phase B 同日完成)
**上游方案:** `RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §4(D0 扩湖:10,000 models × 2,000 datasets)
**最终结论(一句话):** 离线审计(Phase A)后用户判定 130 个 gold 根数据集不够,
追加 dataset-first 在线补采(Phase B):扫描 **6,352** 个候选数据集、补采 **226 数据集
+ 2,447 eval-bearing 模型**元数据后,合并湖达到 **15,210 入湖模型**、
**gold 可评根数据集 211**(非 bitext gold 节点 439)、任务类别覆盖 21 类 gold roots ——
D0 两项硬指标(models ≥10K、gold ≥200)在**根数据集这一最严口径**上也已达标。

---

## 0. 六项必报数字(用户要求,最终合并湖口径)

| # | 指标 | 数字 | 来源 |
|---|---|---|---|
| 1 | raw candidate dataset IDs(在线扫描候选,canon 去重前) | **6,352** | `d0_harvest_seed_raw.csv` |
| 2 | canonical dataset IDs(canon_key 去重后) | **5,639** | `d0_harvest_seed.csv` |
| 3 | distinct root datasets(折叠 config/per-language 后) | **5,637** | 同上 `root` 列 |
| 4 | 具有 ≥2 eval-bearing models 的 roots(邻域实测,去重安全下界) | **475** | 同上(per-dataset max) |
| 5 | **gold roots(≥3 个非恒定标签)** | **211**(Phase A 仅 130) | `d0_root_pool.csv` / funnel `gold_evaluable_roots` |
| 6 | 各任务类别 root 数量 | 见 §5.3 表 | funnel `roots_by_task` |

> 口径说明:#4 是 seed 阶段对每个候选数据集列其模型邻域(`filter=dataset:<id>`,
> cardData 判 model-index)统计的,按 root 取 per-dataset 最大值(跨数据集模型并集
> 只会更大,故为下界);#5 是合并缓存的 model-index 观测按 root 聚合后
> ≥3 distinct 模型且单指标值非恒定(独立按 per-model 先折叠重算得 209,±2 为
> 折叠时点口径差,量级一致)。

## 1. Phase A —— 离线入湖审计(先吃存量)

`hf1000d_2000m/hf_cache/` 是当初 dataset-first harvest 的全量原始缓存:
**12,871 模型 JSON + 4,460 数据集 JSON**(远超入选的 2,000/1,000)。放宽入湖标准
先在存量池上重过滤:

```
intake = 有主指标 model-index 记录(白名单指标 + 数值可归一)
       ∨ 有 lineage 线索(cardData.base_model 或 base_model:* tag)
       ∨ family 可解析(规则/动态推断 ≠ Other)
排除:disabled ∨ private(实际为 0);标签侧丢弃数据集名 canon 后为空/"." 的垃圾观测
```

Phase A 结果(详细漏斗见 git 历史或 `d0_funnel.json` 重跑):模型 12,765 ✅、
gold 节点 499、**但 distinct gold 根数据集仅 130**,且 bitext(tatoeba+ntrex)与
amazon_massive per-language 两类近亲占 gold 节点的 55% —— **任务面不足,
用户判定离线不够,触发 Phase B**。

## 2. Phase B —— 在线 dataset-first 补采(用户指令)

**方向(2026-07-18 用户定调):** 1.2 万模型池可以放弃大部分;必须实质提升的是
**数据集面**和**与数据集紧密耦合的模型**(eval-bearing 邻域)。

### 2.1 采集策略(`d0_online_harvest.py`,三阶段、全缓存、断点续跑)

| 阶段 | 内容 | 网络量 |
|---|---|---|
| `seed` | 候选数据集 = 14 个 task_categories 各 top-300(按 downloads)+ 37 个 benchmark 搜索 + Phase-A 缺元数据根回补(三级解析:KEY_TO_HFID → 直连 id 探测 → 放宽搜索);对**每个候选**列模型邻域(`/api/models?filter=dataset:<id>&cardData=true`,limit 100,稀疏时并查裸名),数 eval-bearing 模型数 | ~6.4K 列表调用,~2h |
| `datasets` | keep 数据集(邻域 ≥2 eval-bearing ∨ Phase-A 缺口)拉全量 JSON | +226 新 JSON,0 失败 |
| `models` | keep 数据集的 eval-bearing 邻域模型拉全量 JSON(每数据集 cap 40,按 downloads;与 Phase-A 缓存去重) | 计划 4,084,新拉 **2,447**,失败 11 |

产出独立落盘 `d0_lake_cache/{list,datasets,models}/`,**冻结的 `hf_cache/` 零写入**。

### 2.2 seed 漏斗

```
6,352 raw 候选 → 5,639 canonical → 5,637 roots
              → keep 530(528 roots;n_eval≥2 或 Phase-A 缺口)
   keep 构成:category 扫描 382 + benchmark 搜索 89 + Phase-A 回补 59
   keep 邻域 eval-bearing 模型:中位数 5,P75 11,max 100
```

## 3. 合并审计(Phase A ∪ Phase B 缓存,`d0_intake_audit.py`)

### 3.1 模型侧

| 阶段 | Phase A | **A+B 合并** |
|---|---|---|
| 缓存模型 JSON | 12,871 | **15,318**(=12,871+2,447,零重复) |
| 通道 A:主指标 model-index | 3,824 | **4,829** |
| 通道 B:lineage 线索 | 4,985 | 6,469 |
| 通道 C:family 可解析 | 12,733 | 15,166 |
| **intake 并集** | 12,765 | **15,210** ✅ |
| 严格口径(A∨B) | 7,561 | **9,491** |

清单 sha256:`96a3e12a438378484c836f13e5789a9a7f3352b6419c1f87573eea60a523441e`

> 湖构成决策(承用户"模型可弃大部分"):最终建图湖不必用满 15,210。
> **dataset-耦合核心 = 通道 A 的 4,829 个带标签模型 + 其 lineage 邻接**;
> 严格口径 9,491 已近 10K。`d0_model_intake.csv` 三通道 flag 齐全,
> Phase C 建图时按需切换,10K 目标在两种口径下都可满足。

### 3.2 标签侧

| 指标 | 现 2K 湖 | Phase A | **A+B 合并** |
|---|---|---|---|
| distinct (node, model) 标签对 | 7,056 | 53,971 | **56,311**(8.0×) |
| 带标签数据集节点 | 72 有边/362 | 1,395 | **1,754** |
| 节点 ≥3 标签 | — | 988 | **1,118** |
| gold 可评节点 | 61 | 499 | **613** |
| gold 可评 coarse 数据集 | — | 152 | **225** |
| **gold 可评根数据集(最严口径)** | ~10–20* | 130 | **211** ✅ |
| 带标签 roots | — | — | 497 |

\* 现 2K 湖的 61 个可评按根折叠后约 10–20(大量 glue/tweet_eval 亲缘)。

### 3.3 集中度(诚实口径)

| 拆解 | Phase A | A+B |
|---|---|---|
| gold 节点 | 499 | 613 |
| bitext 类节点 | 174(35%) | 174(**28%,稀释中**) |
| 非 bitext gold 节点 | 325 | **439** |
| gold 节点的 distinct roots | 194* | 194 |
| 非 bitext distinct roots | 124 | **188** |

\* 194 = gold **节点**的根集合;211 = 按根**重新聚合**后判 gold(根内跨 config 汇总
可让单节点不足 3 标签的根达标),后者是 #5 的正式口径。
top 膨胀根不变(tatoeba 112、ntrex 54、massive×2 102)——**评测报表必须按根分层**。

## 4. 数据集节点规模(≥1,500–2,000 目标)

带标签节点 1,754 + 合并数据集元数据缓存 4,640(4,460 旧 + 226 新,含 gold 的
573/613)→ 2,000 节点选择空间充足;仍缺元数据的 gold 节点 40 个
(`d0_missing_dataset_metadata.csv`,下一轮轻量补采)。

## 5. 六项数字的支撑表

### 5.1–5.2 seed 漏斗与 roots 见 §0、§2.2

### 5.3 各任务类别 root 数量(funnel `roots_by_task`)

| dominant_task | 带标签 roots | gold roots |
|---|---|---|
| text-classification | 86 | 44 |
| token-classification | 64 | **39** |
| text-generation | 63 | 22 |
| classification(MTEB 系) | 54 | 34 |
| sts | 30 | 24 |
| question-answering | 19 | 8 |
| reranking | 18 | 12 |
| bitextmining | 9 | 7 |
| summarization | 9 | 2 |
| multiple-choice | 8 | — |
| fill-mask | 7 | — |
| NLI(三种写法合计) | 8 | 3 |
| 其余(image/audio/code/math/翻译等 long tail) | 47 | 16 |
| **合计** | **422** | **211** |

> 合计 422 是**有 bounded 指标标签**的 roots(gold 判定的分母);any-label roots
> 为 497(含 wer/cer/bleu 等非 bounded 指标的 75 个根)。

对比 Phase A:gold roots 里 token-classification 从个位数升到 39、reranking 12、
sts 24 —— **不再是 bitext/intent 双寡头**,21 个任务类别有 gold root。

## 6. 复现

```bash
cd ModelLakeFishing/stage1BuildTransferGraph
# 1) 在线补采(结果已全部落盘缓存,重跑=校验,不产生新网络请求)
../.venv/Scripts/python.exe d0_online_harvest.py --phase all
# 2) 合并审计(零网络,~2 分钟,重写 artifacts/d0_lake/ 全部产物)
../.venv/Scripts/python.exe d0_intake_audit.py
```

六项数字逐一断言(全部通过):

```python
import json, pandas as pd
base = r'ModelLakeFishing\stage1BuildTransferGraph\artifacts\d0_lake'
f = json.load(open(base + r'\d0_funnel.json'))
hs = f['harvest_seed']
assert hs['raw_candidate_dataset_ids'] == 6352          # 1 raw 候选
assert hs['canonical_dataset_ids'] == 5639              # 2 canonical
assert hs['distinct_root_datasets'] == 5637             # 3 roots
assert hs['roots_ge2_eval_bearing_models'] == 475       # 4 ≥2 eval-bearing
assert f['gold_evaluable_roots'] == 211                 # 5 gold roots
assert sum(f['roots_by_task']['gold'].values()) == 211  # 6 任务类别表自洽
assert f['intake_union'] == 15210 and f['gold_evaluable_nodes'] == 613

# 独立复核 #5(不经 funnel,直接从观测表)
r = pd.read_csv(base + r'\d0_root_pool.csv')
assert int(r['gold_evaluable'].sum()) == 211
# 更严的 per-model 先折叠口径:
o = pd.read_parquet(base + r'\d0_observations.parquet')
BH = {'accuracy','f1','matthews_correlation','pearson','spearman','exact_match','map','mrr','rougeL'}
b = o[o['metric_canonical'].isin(BH)].copy()
b['root'] = b['dataset_canonical'].str.split('/').str[0]
cov = b.groupby(['root','metric_canonical'])['model_id'].nunique().reset_index(name='n')
best = cov.sort_values(['root','n','metric_canonical'],ascending=[True,False,True]).drop_duplicates('root')
sel = b.merge(best[['root','metric_canonical']],on=['root','metric_canonical'])
g = sel.groupby(['root','model_id'])['value'].median().reset_index()
st = g.groupby('root').agg(n=('model_id','nunique'),nu=('value','nunique'))
print(int(((st['n']>=3)&(st['nu']>1)).sum()))           # 209(±2 口径差,见 §0)
```

## 7. 改动清单(两阶段累计)

| 文件 | 性质 |
|---|---|
| `stage1BuildTransferGraph/d0_intake_audit.py` | 新增;后扩展为双缓存合并 + root 粒度 gold + 任务分布 + seed 汇总 |
| `stage1BuildTransferGraph/d0_online_harvest.py` | 新增;dataset-first 在线补采三阶段 |
| `stage1BuildTransferGraph/d0_lake_cache/{list,datasets,models}/` | 新缓存(6.4K 列表 + 226 数据集 + 2,447 模型 JSON);冻结 `hf_cache/` 零写入 |
| `artifacts/d0_lake/d0_funnel.{json,md}` | 漏斗机读版(本文全部数字来源) |
| `artifacts/d0_lake/d0_model_intake.csv` | 15,318 模型 × 三通道 flag |
| `artifacts/d0_lake/d0_observations.parquet` | 107,661 canonical 观测(去重纪律:(node, model, metric) 单行) |
| `artifacts/d0_lake/d0_dataset_pool.csv` / `d0_root_pool.csv` | 节点/根两粒度标签统计 + gold 判定 |
| `artifacts/d0_lake/d0_harvest_seed{,_raw}.csv` | 在线候选扫描(#1–#4 的来源) |
| `artifacts/d0_lake/d0_intake_models.json` (+sha256) | 冻结候选清单 15,210 |
| `artifacts/d0_lake/d0_missing_dataset_metadata.csv` | 仍缺元数据的 40 gold 节点 |

## 8. 下一步(D0 剩余阶段)

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **C. 湖定稿 + 特征离线构建** | 模型湖口径定稿(dataset-耦合核心 4,829 + lineage vs 严格 9,491 vs 全量 15,210);2,000 数据集节点选择(gold 613 节点全入 + 任务分层补充);xd0 gpt-neo 嵌入(GPU)、xm0、e_task vocab | 本阶段产物 |
| **D. 建图 + 双契约** | trained_on 直接从 `d0_observations.parquet` 生成(源头去重);Stage-1 sanity + Stage-2 契约 | C |
| **E. 训练侧准备** | pyg-lib/torch-sparse 安装;B0/G2 基线重跑 + 三列基线 + conditioning gain(**按根分层报告**) | D |
| **F. Stage-3 重索引** | export → build → query,fidelity ≥0.99 | E |

G-D0 验收对照:models ✅(15,210 / 严格 9,491)/ gold ≥200 ✅(节点 613、
coarse 225、**根 211** —— 三口径全过)/ 双契约、基线、重索引待 C–F。
