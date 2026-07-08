# RESULTS — 每个数字 + 复现代码

本文件把 Kendall guide 实验里报告过的**每一个数字**列出，并附上**得到该数字的确切命令/代码**。
所有数字均来自 `artifacts/ablation/<name>.json`（由 `ablation.py` 产出），可直接重跑或从 JSON 重读。

## 0. 通用环境

```bash
# 工作目录（所有命令的 CWD）
cd d:/research/model_lake/codes
# venv python（CUDA，约 2s/epoch）
PY=ModelLakeFishing/.venv/Scripts/python.exe
# 图文件（2000 models, 362 datasets）
GRAPH=ModelLakeFishing/stage1BuildTransferGraph/hgraph_hf1000d_2000m_xm0_xd0.pt
```

每个配置统一用 3 个固定 split seed（0,1,2）、1 个 init seed、25 epoch（除注明）。
**从 JSON 重读任意聚合数字的通用代码：**

```python
import json
a = json.load(open("ModelLakeFishing/stage2TrainGraphSAGE/artifacts/ablation/<name>.json"))
a["aggregate"]["tau_macro"]      # -> [mean, std]
a["aggregate"]["mean_cos"]       # -> [mean, std]
a["head_aggregate"]["hit@10"]    # -> [mean, std]；同理 recall_top3@10 / ndcg@50 / regret@10
[r["tau_macro"] for r in a["runs"]]   # -> 逐 split 的 [s0, s1, s2]
```

---

## 1. 全部聚合数字总表（tau_macro mean±std / mean_cos / head retrieval）

| name | tau_macro | mean_cos | hit@10 | recall_top3@10 | ndcg@50 | regret@10 |
|---|---|---|---|---|---|---|
| B0 | 0.1928 ± 0.059 | 0.4403 | 0.809 | 0.796 | 0.840 | 0.0055 |
| B1 | 0.2121 ± 0.027 | 0.3400 | 0.785 | 0.777 | 0.840 | 0.0109 |
| B2 | 0.1977 ± 0.064 | 0.3871 | 0.801 | 0.769 | 0.840 | 0.0080 |
| B2e_ctrl | 0.2539 ± 0.054 | 0.3911 | 0.817 | 0.812 | 0.843 | 0.0101 |
| B3_sim | 0.2316 ± 0.060 | 0.3944 | 0.825 | 0.804 | 0.841 | 0.0102 |
| B3_simtr | 0.2664 ± 0.046 | 0.3916 | 0.842 | 0.809 | 0.842 | 0.0084 |
| B3_all | 0.2790 ± 0.022 | 0.3711 | 0.785 | 0.772 | 0.842 | 0.0104 |
| B4_grouped | 0.1746 ± 0.025 | 0.0704 | 0.738 | 0.743 | 0.836 | 0.0200 |
| B6_heads | 0.1722 ± 0.021 | 0.3971 | 0.786 | 0.756 | 0.838 | 0.0099 |
| BEST | 0.1826 ± 0.036 | 0.4192 | 0.738 | 0.757 | 0.834 | 0.0102 |
| **B5_ranknet** | **0.3982 ± 0.035** | 0.3482 | 0.850 | 0.841 | 0.848 | 0.0032 |
| R_tohet | 0.3153 ± 0.031 | 0.3346 | 0.857 | 0.827 | 0.844 | 0.0044 |
| R_weights | 0.3541 ± 0.045 | 0.3381 | 0.857 | 0.844 | 0.846 | 0.0047 |
| R_heads | 0.3826 ± 0.013 | 0.3691 | 0.849 | 0.825 | 0.847 | 0.0035 |
| R_mg00 | 0.3632 ± 0.068 | 0.3427 | 0.889 | 0.873 | 0.848 | 0.0031 |
| **R_mg02** | **0.4029 ± 0.022** | 0.3047 | 0.857 | 0.843 | 0.849 | 0.0031 |
| P6_dm05 | 0.3425 ± 0.065 | 0.2550 | 0.905 | 0.889 | 0.846 | 0.0032 |
| **P6_dm10** | 0.3277 ± 0.070 | 0.2182 | **0.905** | **0.902** | 0.847 | 0.0019 |
| P7_es | 0.3699 ± 0.042 | 0.3465 | 0.904 | 0.862 | 0.846 | 0.0024 |
| P7_lr3e3 | 0.3705 ± 0.022 | 0.3054 | 0.849 | 0.825 | 0.844 | 0.0046 |
| ACCEPTED_ranknet (seed0) | 0.3957 (1 seed) | 0.3495 | 0.833 | 0.802 | 0.848 | 0.0059 |

---

## 2. 每个数字的生成命令

下面每条命令产出对应 `<name>.json`，表中该行的所有数字即从该 JSON 读出（见 §0 重读代码）。

### Phase 0 — 基线 B0
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B0 --seeds 3 --epochs 25
# -> tau_macro 0.1928±0.059, mean_cos 0.4403, hit@10 0.809, regret@10 0.0055
```

### Phase 1 — dataset 图（B1 删除 / B2 top-k）
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B1 --similar_to_mode none --seeds 3 --epochs 25
# -> tau_macro 0.2121±0.027, mean_cos 0.3400
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B2 --similar_to_mode topk_unweighted --similar_to_k 10 --seeds 3 --epochs 25
# -> tau_macro 0.1977±0.064, mean_cos 0.3871
```

### Phase 2 — edge-aware（控制组 + 逐步加权）
```bash
A="--similar_to_mode topk_unweighted --similar_to_k 10 --seeds 3 --epochs 25 --edge_aware"
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B2e_ctrl $A --weighted_relations none
# -> tau_macro 0.2539±0.054   （edge-aware 结构、无权重）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B3_sim   $A --weighted_relations "similar_to"
# -> tau_macro 0.2316±0.060
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B3_simtr $A --weighted_relations "similar_to,trained_on,rev_trained_on"
# -> tau_macro 0.2664±0.046
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B3_all   $A
# -> tau_macro 0.2790±0.022   （全部关系加权，含 lineage）
```

### Phase 3/5/“kitchen-sink” — B4_grouped / B6_heads / BEST
```bash
B="--similar_to_mode topk_unweighted --similar_to_k 10 --seeds 3 --epochs 25 --edge_aware --weighted_relations none"
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B4_grouped $B --grouped
# -> tau_macro 0.1746±0.025, mean_cos 0.0704   （被否：全图坍塌）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B6_heads $B --separate_heads
# -> tau_macro 0.1722±0.021   （被否）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name BEST --similar_to_mode topk_unweighted --similar_to_k 10 \
    --seeds 3 --epochs 25 --edge_aware --rank_loss ranknet --rank_min_gap 0.01 --separate_heads --grouped
# -> tau_macro 0.1826±0.036   （被 grouped+heads 拖累）
```

### Phase 4 — RankNet（主胜负手 B5）+ 分解 + min_gap 扫描
```bash
RK="--seeds 3 --epochs 25 --rank_loss ranknet"
# B5：edge-aware 无权重 + RankNet min_gap 0.01（主结果）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name B5_ranknet $RK --rank_min_gap 0.01 \
    --similar_to_mode topk_unweighted --edge_aware --weighted_relations none
# -> tau_macro 0.3982±0.035, hit@10 0.850, regret@10 0.0032
# R_tohet：RankNet 不用 edge-aware（隔离损失贡献）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name R_tohet $RK --rank_min_gap 0.01 --similar_to_mode topk_unweighted
# -> tau_macro 0.3153±0.031   （仅换损失 = +0.12）
# R_weights：RankNet + 全部边权（验证权重在 RankNet 下是否帮助）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name R_weights $RK --rank_min_gap 0.01 --similar_to_mode topk_unweighted --edge_aware
# -> tau_macro 0.3541±0.045   （加权反而下降）
# R_heads：RankNet + 分离头
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name R_heads $RK --rank_min_gap 0.01 --similar_to_mode topk_unweighted --edge_aware --weighted_relations none --separate_heads
# -> tau_macro 0.3826±0.013
# min_gap 扫描（edge-aware 无权重 + RankNet）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name R_mg00 $RK --rank_min_gap 0.0  --similar_to_mode topk_unweighted --edge_aware --weighted_relations none
# -> tau_macro 0.3632±0.068   （min_gap=0：最噪）
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name R_mg02 $RK --rank_min_gap 0.02 --similar_to_mode topk_unweighted --edge_aware --weighted_relations none
# -> tau_macro 0.4029±0.022   （min_gap=0.02：最佳最稳）
```

### Phase 6 — dataset→model 对比
```bash
B5="--seeds 3 --epochs 25 --rank_loss ranknet --rank_min_gap 0.01 --similar_to_mode topk_unweighted --edge_aware --weighted_relations none"
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name P6_dm05 $B5 --lambda_dm_contrast 0.5
# -> tau_macro 0.3425±0.065, hit@10 0.905, recall_top3@10 0.889
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name P6_dm10 $B5 --lambda_dm_contrast 1.0
# -> tau_macro 0.3277±0.070, hit@10 0.905, recall_top3@10 0.902   （服务侧最佳）
```

### Phase 7 — 早停 + lr
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name P7_es --seeds 3 --epochs 40 \
    --rank_loss ranknet --rank_min_gap 0.01 --similar_to_mode topk_unweighted --edge_aware --weighted_relations none \
    --early_stop --patience 10
# -> tau_macro 0.3699±0.042, hit@10 0.904
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name P7_lr3e3 $B5 --lr 0.003 --epochs 40
# -> tau_macro 0.3705±0.022   （方差最低）
```

### Accepted 候选 checkpoint（roundtrip 验证）
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.ablation --name ACCEPTED_ranknet \
    --similar_to_mode topk_unweighted --edge_aware --weighted_relations none \
    --rank_loss ranknet --rank_min_gap 0.02 --seeds 1 --epochs 25 \
    --save_ckpt ModelLakeFishing/stage2TrainGraphSAGE/artifacts/ablation/stage2_hf1000d_ranknet_candidate.pt
# -> seed0 tau_macro 0.3957, hit@10 0.833; "[ckpt] saved ... -> OK (roundtrip matches)"
```

---

## 3. 逐 split tau_macro（[s0, s1, s2]）

读取代码：`[r["tau_macro"] for r in json.load(open("<name>.json"))["runs"]]`

| name | s0 | s1 | s2 |
|---|---|---|---|
| B0 | 0.1371 | 0.2749 | 0.1666 |
| B1 | 0.1844 | 0.2031 | 0.2488 |
| B2 | 0.1845 | 0.2819 | 0.1268 |
| B2e_ctrl | 0.1927 | 0.3231 | 0.2459 |
| B3_all | 0.2519 | 0.2803 | 0.3047 |
| B5_ranknet | 0.3509 | 0.4357 | 0.4080 |
| R_mg02 | 0.3957 | 0.4323 | 0.3808 |
| P6_dm10 | 0.3138 | 0.4194 | 0.2500 |

（B5_ranknet / R_mg02 在 3 个 split 上**全部**高于 B0 同 split → 满足 guide “在所有固定 split 一致复现”的条件。）

---

## 4. 配对 bootstrap 比较（门槛判定）

命令：`$PY -m ModelLakeFishing.stage2TrainGraphSAGE.compare --base <A> --cand <B>`
（在相同 (split,dataset) tag 上配对，10000 次 bootstrap，seed 固定 → 数字确定可复现）

| 比较 | Δtau_macro | 95% CI | P(>0) | 判定 |
|---|---|---|---|---|
| `--base B0 --cand B5_ranknet` | **+0.2051** | **[+0.1218, +0.2943]** | **1.00** | **排除 0 → 提升** |
| `--base B0 --cand B2e_ctrl` | +0.0609 | [−0.0203, +0.1411] | 0.93 | 含 0（但 3 split 全升）|
| `--base B0 --cand B3_all` | +0.0857 | [−0.0005, +0.1739] | 0.97 | 含 0（3 split 全升）|
| `--base B5_ranknet --cand P6_dm10` | −0.0698 | [−0.1367, −0.0039] | 0.02 | 排除 0（tau 显著下降，但 head↑）|

head retrieval 的逐项退化检查也由同一 `compare.py` 输出（B0→B5 全部不退化；B5→P6_dm10：hit@10 +0.055、recall_top3 +0.060、ndcg@50 −0.0006）。

---

## 5. guide 的“confirmed facts”基线数字（生产 checkpoint 诊断）

来源文件：`artifacts/hf1000d_2000m/stage2_hf1000d_2000m_diagnostics.json`
生成命令：
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.diverse_diagnostics \
    --pt $GRAPH --ckpt ModelLakeFishing/stage2TrainGraphSAGE/artifacts/hf1000d_2000m/stage2_hf1000d_2000m_candidate.pt
```
读取代码：`json.load(open(".../stage2_hf1000d_2000m_diagnostics.json"))["raw"|"trained"][<key>]`

| 数字 | 值 | JSON 键 |
|---|---|---|
| model / dataset 节点 | 2000 / 362 | raw.n_models / raw.n_datasets |
| trained_on / similar_to / lineage 边 | 12205 / 130680 / 96 | raw.edges[...] |
| dataset 相似度有效秩（perf-bearing）| 10.27 | raw.dataset_sim_eff_rank_perfbearing |
| z_m / z_d 参与率 | 2.741 / 2.413 | trained.z_m_participation_ratio / z_d_… |
| seed-0 诊断 tau_macro / 可评分 dataset | 0.1653 / 42 | trained.tau_macro / n_datasets_scored |
| mean pairwise cosine | 0.5028 | trained.mean_pairwise_cosine |

新 accepted checkpoint 上的同一诊断：
```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.diverse_diagnostics \
    --pt $GRAPH --ckpt ModelLakeFishing/stage2TrainGraphSAGE/artifacts/ablation/stage2_hf1000d_ranknet_candidate.pt
# -> tau_macro 0.3729, head_retrieval.hit@10 0.8333, ndcg@50 0.8463（seed-0 单点，与 R_mg02 多 seed 0.40 一致）
```

---

## 6. 测试（每组测试的运行命令）

```bash
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase1_graph         # 图度数/无自环/top-k 选择
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase2_edge_aware    # edge_attr 改变输出/置零关系/roundtrip
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase3_macro         # 宏平衡：逐 dataset 均值与 pair 数无关
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.tests.test_phase4_5_loss_heads  # RankNet 排序方向/min_gap/分离头 MIPS 等价
# 全部输出 "... OK"
```

---

## 7. 一键复现全部（可选）

```bash
# 逐条运行 §2 的所有命令即可重建全部 <name>.json；随后：
$PY -m ModelLakeFishing.stage2TrainGraphSAGE.compare --base B0 --cand B5_ranknet   # 复现 §4 门槛数字
# 数字会因 CUDA 非确定性有 ~±0.01 抖动，但结论（B5 通过严格门槛、min_gap 0.02 最优、grouped/heads 被否）稳定。
```
