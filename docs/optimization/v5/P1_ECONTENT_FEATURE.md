# v5 e_content 特征化可审计记录 —— MiniLM 编码样本文本 → xd0 内容视图 → 组合图

**日期:** 2026-07-19
**上游方案:** `../RETRIEVAL_QUALITY_REMEDIATION_PLAN.md` §0.3 P1 / §0.7(P3 并入)
**前置:** P1 采样已落盘(`P3_P1_HARVEST.md`,312/1,463 has_content);
P3 D0.5 湖已随审计重跑并入 intake/observations/pool。
**本步:** 把 P1 的样本文本编码为 e_content 视图拼入 xd0,**一次建成 D0.5+content
组合图**(P3 §0.7 决策:不单独重基线,合并验证)。

---

## 1. 特征形态

```
xd0(v4 D0)   = [ e_name 64 ‖ e_card 384 ‖ e_stats 10 ]              = 458
xd0(D0.5+content) = [ e_name 64 ‖ e_card 384 ‖ e_content 384 ‖ e_stats 10 ] = 842
e_content(node) = MiniLM(样本文本)     若 has_content
                = 0 向量(384)          若无内容(显式缺失)
e_stats 第 8 槽 = has_content 二值标志(让 GNN 门控内容块)
```

**缺失处理的设计(与方案"回退元数据串"一致):** 无内容节点的 e_content 置零,
但其 **e_name/e_card/e_stats 元数据视图完全保留**——即"回退到纯元数据"是
通过"内容块置零、元数据块不变"实现的,不是把元数据复制进内容块(那会让
两块冗余、污染消融)。has_content 标志同时作为特征进 e_stats,让模型学会
"仅在有内容时使用内容块"。

## 2. 代码改动

| 文件 | 改动 |
|---|---|
| `stage1BuildTransferGraph/d0_build_graph.py` | 加 `--content` 标志:`load_content_texts`(复用 p1 的 `_safe_name` 按节点名对齐,mappedID 无关)→ MiniLM 编码 has_content 节点 → e_content 视图插在 e_card 与 e_stats 之间;`has_content` 进 e_stats 第 8 槽 + 挂 `data['dataset'].has_content`(随子图切片,供分层消融);`view_dims`/契约维度/report 随标志切换。**默认 OFF 逐位复现 v4 图。** 输出独立路径 `hgraph_d0_v1_content.pt`(不覆盖已上线的 `hgraph_d0_v1.pt`) |
| `stage2TrainGraphSAGE/w1_dzero.py` | 加 `--graph` 参数,使组合图重基线可直接复用现有 runner(默认仍指 v4 D0 图) |

## 3. 组合图产出(`hgraph_d0_v1_content.pt`)

```
sha256: dea91542bd0cf6d900220819dcb978cf61a1d0bbde7c9bcbb513a4b98b78c1f3
models 10,234(D0.5,v4 为 9,491)| dataset nodes 1,514(v4 1,463)
roots 461(v4 421)| trained_on 55,140 | lineage 1,033 | similar_to 30,280
task vocab 19(v4 17,P3 新增 2 类)| gold 617
xd0 frozen 842(+e_content 384)
content 覆盖:全局 20.6% | gold 节点 26.6%
```

**P3(D0.5 扩容)与 P1(内容)在这一张图里同时生效**:湖从 9,491→10,234、
根 421→461(P3),xd0 加 e_content(P1)。v4 的 `hgraph_d0_v1.pt`(L1L3b 已
上线的图)保持不动,两图并存。

## 4. 验证

- **端到端训练**:`HeteroGraphSAGE(dataset_in_dim=842)` 前向 OK,
  z_m [10234×128] / z_d [1514×128];DatasetNodeEncoder 自动吃 842 维;
- **切片对齐**:`data.subgraph({dataset: 30})` 后 `has_content` 长度 30,
  随子图对齐(供 loader mini-batch 分层);
- **1-epoch 冒烟**(L1L3b @ 组合图,seed 0):跑通,lake logQ
  models=10,234 labeled=3,934 max_deg=224——D0.5 度数分布(v4 D0 s0 为 269),
  α=0.75 温度化正常;
- **契约**:5 边型、mappedID 行序、attr∈[0,1]、task/family Other=0 断言全过。

## 5. 复现

```bash
cd ModelLakeFishing/stage1BuildTransferGraph
../.venv/Scripts/python.exe d0_build_graph.py --content    # -> hgraph_d0_v1_content.pt
```

复核:

```python
import torch, json
p = torch.load(r'ModelLakeFishing\stage1BuildTransferGraph\hgraph_d0_v1_content.pt',
               map_location='cpu', weights_only=False)
assert p['data']['dataset'].x.shape[1] == 842
assert p['xd0_meta']['view_dims']['e_content'] == 384
assert p['xd0_meta']['has_content_view'] is True
hc = p['data']['dataset'].has_content
assert 0.20 < float(hc.mean()) < 0.22
r = json.load(open(r'ModelLakeFishing\stage1BuildTransferGraph\artifacts\d0_lake\d0_graph_report_content.json'))
assert r['models'] == 10234 and r['roots'] == 461 and r['dataset_frozen_dim'] == 842
```

## 6. 8-seed 分层终裁(2026-07-19)—— P1 内容不显著,承诺未达成

**实验设计(干净隔离内容):** 控制图 `hgraph_d05_nocontent.pt` = 内容图逐位
复制、仅 e_content 块(842 维 [448:832])+ has_content 标志位归零。节点/边/
mappedID/split 完全相同 → 同 seed 同测试集,逐数据集配对精确,内容是唯一变量
(P3 湖扩容在两图都在,被 hold 住)。L1L3b × 8 seeds × 两图 = 16 训练。
脚本 `v5_verdict.py`,产物 `v5_verdict.{json,md}`。

**root_macro(公平口径,8 seeds):**

| 分层 | n_ds/seed | gold@10 内容 | gold@10 对照 | gold-gap@10 内容 | gold-gap@10 对照 |
|---|---|---|---|---|---|
| all | 133 | 0.134±0.118 | 0.131±0.128 | 0.211 | 0.236 |
| **has_content** | 32 | **0.170±0.142** | 0.141±0.126 | 0.237 | 0.276 |
| gold_has_content | 31 | 0.163 | 0.143 | 0.230 | 0.270 |
| no_content(对照层) | 100 | 0.090 | 0.120 | 0.172 | 0.196 |

**裁决:REJECT —— 内容无可靠增量。**

1. **has_content 层 gold@10 名义 +0.029(0.170 vs 0.141),但 per-seed 仅 4/8
   上升(掷硬币),且 gold-gap@10 反向(0.237 < 0.276)。** 两个主指标符号打架
   + per-seed 无一致性 = 噪声,不是效应。
2. 唯一"排除 0"的 CI(flat gold-gap@10 all)是 **big-sibling-family 的 flat
   假象**(per-dataset 池化被兄弟众多的大根族主导)——正是 root 口径(W1 教训)
   要中和的东西;root_macro 下内容反而略低。
3. no_content 对照层 gold@10 内容 0.090 < 对照 0.120——内容图为无内容节点
   徒增了内容块参数(全零输入),反而添了噪声。

**§0.6 承诺检查:未达成。** 内容图 root gold@10 = **0.134**(目标 0.22–0.29,
**仅关闭 0.100→0.563 差距的 7%**);root gold-gap@10 = 0.211(目标 ≥0.45)。

**诚实诊断(为何失败):**
- **覆盖 21% → 内容视图 79% 是零**,GNN 学会忽略它;
- 即使有内容的节点,MiniLM(样本文本)对 gold 区分未见超出 e_card(元数据)
  的信息——数据集"长什么样"和"叫什么/什么任务"高度相关,内容视图与元数据
  视图冗余;
- 8 seeds × 32 has_content 数据集功效不足以分辨 <0.05 的效应(E10 老问题)。

**顺带的 P3 观察(未确证):** D0.5-nocontent root gold@10 = 0.131,名义高于
v4 的 0.100——但两图测试根群体不同(不是配对),不能归因为 P3 的净效应,
只能说"更多根未见伤害,可能小幅有利"。

**结论:** L1L3b @ `hgraph_d0_v1.pt` 维持晋升配置(索引仍在服务);
P1 内容与 P3 扩容(现幅度)均**不进入服务栈**。O-sibling 0.563 的头顶空间
**在 21% 内容覆盖下无法由内容通道兑现**——真正有 headroom 的是 sibling-label
通道(P5 根级池化,bitext 家族的正解),或需要覆盖远高于 21% 的内容采集
(更强的采样源,非 datasets-server)。这两条是 v5 之后的方向。