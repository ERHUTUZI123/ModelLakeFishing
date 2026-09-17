# X6：3M 候选池基线（执行计划与记录）

X6 为论文的 3M-scale 主结果提供最小而充分的参照。被评方法是 [`X4.md`](X4.md) 的 GD 合并臂，
查询口径是 [`X5.md`](X5.md) 的 `gold_eligible` 过滤；候选池、划分和 gold 均沿用
[`1Mplan.md`](1Mplan.md) 的 `rf-gold-2.0`，本阶段不重新定义任务。

## 1. 要回答的问题

只回答三个会改变论文结论的问题：

1. **查询相关性带来什么？** 与查询无关的热度排序，和使用查询文本的稀疏/稠密检索相比怎样。
2. **监督训练带来什么？** 冻结语义检索和使用同一训练目标的无图双塔相比怎样。
3. **图结构带来什么？** 无图双塔和完整 GraphSAGE 在其余训练条件相同的情况下相比怎样。

不做点赞、模型大小、发布时间、关系传播、RRF 组合，也不为 ModelLens、TransferGraph 或逐模型前向式
方法建设大规模外推实验。这些实验不能干净分离上述三项，且不会改变本论文对 3M 候选池的核心比较。

## 2. 最小基线集

随机排序只作解析参照 `10/N`，不占一条实验基线。正式基线共四条：

| ID | 基线 | 查询相关 | 训练 | 图 | 作用 |
|---|---|:---:|:---:|:---:|---|
| P | Popularity：按快照 `downloads` 排序 | 否 | 否 | 否 | 查询无关的实用 sanity check |
| L | BM25：查询为 dataset--task，文档为 model id、tags、pipeline tag | 是 | 否 | 否 | 稀疏词法检索 |
| S | Frozen MiniLM：`cos(x_d[:,64:448], x_m[:,64:448])` | 是 | 否 | 否 | 稠密语义检索，不读监督统计 |
| T | NoGraph dual tower：与 X4-GD 相同的输入、损失、采样、超参和三 seed，仅令消息传递层数为 0 | 是 | 是 | 否 | 隔离监督训练，并作为图结构的直接对照 |

完整方法 G 不是基线；它是 X4-GD 的既有结果。解释链为 `P -> {L,S} -> T -> G`。
L 和 S 都保留，因为它们代表互补的词法与语义查询匹配，且 L 可以利用模型卡中明确的数据集标签。

## 3. 统一口径

| 项 | 固定值 |
|---|---|
| 候选池 | 全部 3,016,439 个模型；缺少某项属性时赋该基线最低分，不移出候选池 |
| 查询集 | X5 过滤后，seed 0/1/2 分别为 1,476 / 1,101 / 1,545 |
| split seed | 0 / 1 / 2，与 X4-GD 相同；`init_seed=0` |
| 主指标 | `gold@10`；同时报告 `gold@1`、`top3@10`、`root_gold@10`、median gold rank |
| harness | `scale/global_metrics.py` 的同一套 rank 与 aggregate 实现 |
| 完整方法 G | `gold@10` 0.1355 / 0.1417 / 0.1508，均值 0.1427 |

训练自由基线不能沿用“并列块第一名”的乐观解释。P/L/S 统一用一个由 `mappedID`
生成的固定双射哈希作为次级排序键；主分数不同时它不起作用，主分数相同时给出唯一、与标签无关且可复现的顺序。
T/G 保留 X4 已审计的稠密嵌入 harness；该 harness 仍对主分数使用严格大于计数，以保证 G 与 X4 的已发布数字完全一致。
随机参照仍为解析值 `10 / 3,016,439 = 3.315e-6`。

## 4. 必须通过的代码门

这些检查由代码断言执行，失败时不写最终结果：

- `n_models == 3_016_439`，`mappedID == 0..N-1`；三个 seed 的查询数逐项等于 §3。
- 三个导出目录的 model/dataset 映射完全一致；P/L/S 共用同一个固定 tie-break。
- P/L 的 sidecar 在全部 3,003,759 个快照行上逐行核对 model id；12,680 个历史模型只填缺失值。
- S 的切片严格为 `[64:448]`，不读取监督构造的 `x_d[:,448:]`。
- T 的 resolved config 与 X4-GD 除 `num_layers: 1 -> 0` 和运行标识外相同；运行时断言 `gnn is None`。
- G 与 T 都只使用 held-out 导出的 `z_m_eval` / `z_d_eval`；每个 score 向量长度等于 N。

## 5. 证伪条件与允许的结论

判据在完整重跑前固定：

- 若 `max(L,S) >= G`，论文不得声称训练后的检索表示提高了 3M 排序精度；应报告与最强训练自由基线
  持平或落后，并保留效率结果。
- 若 `T <= max(L,S)`，不得声称监督训练本身带来正增量。
- 若 `G <= T`，不得声称图结构带来正增量。
- 只有当 `G > T > max(L,S)` 时，才能写“监督训练和图结构均带来增量”。不根据幅度另设事后阈值。

P 是查询无关检查：若 P 很强则照实报告，说明 benchmark 与模型热度相关；不更换查询集。

## 6. 执行顺序

1. 修正 tie 处理并运行 P/L/S 的三个 seed；先确认最强训练自由基线。
2. 实现 `num_layers=0`，跑单元测试和 1-epoch smoke；随后按 X4-GD 配方训练 T 的三个 seed。
3. 用现有 `export_rf` 导出 T 的 held-out 嵌入，以同一 X6 evaluator 汇总 P/L/S/T/G。
4. 若第 1 步已有基线超过 G，仍执行 T，因为 T 是图结构增量的唯一直接对照；取消其余扩展实验。

## 7. 运行命令

```powershell
# 本地：sidecar、测试、训练自由基线
.venv\Scripts\python.exe -m scale1m.baseline_sidecar
.venv\Scripts\python.exe -m pytest scale1m/tests/test_x6.py stage2TrainGraphSAGE/tests/test_t0_scale.py -q
.venv\Scripts\python.exe -m scale1m.eval_x6 --stage training-free --seeds 0 1 2

# 训练端：T，三个 seed；实际远端命令与 job id 在结果节记录
python -m scale1m.train_rung --rung full --graph "$DATA_ROOT/data1m/graphs/hgraph_rf" \
  --out "$OUTPUT_ROOT/runs/X6NoGraph_full_s${SEED}_e25" --seed "$SEED" --epochs 25 \
  --family-vocab "$DATA_ROOT/data1m/feats_rf/family_vocab.csv" \
  --fanout --sparse-M --contrast-n-neg 256 --chunked-infer 50000 \
  --skip-diagnostics --lake-gamma 0.5 --global-n-datasets 128 --num-layers 0

# 本地：T 的每个 checkpoint 复用现有导出，再统一汇总
.venv\Scripts\python.exe -m scale1m.export_rf --run <T_RUN> --stage embed --out <T_EXPORT>
.venv\Scripts\python.exe -m scale1m.eval_x6 --stage learned \
  --nograph-exports <T_EXPORT_ROOT> --nograph-run-fmt "X6NoGraph_full_s%d_e25"
```

## 8. 结果与执行记录

训练自由部分已于 2026-09-04 完成。主指标如下；其他已验证指标保存在
[`X6_BASELINES.training_free.json`](X6_runs/X6_BASELINES.training_free.json)。

| 方法 | seed 0 | seed 1 | seed 2 | 均值 `gold@10` |
|---|---:|---:|---:|---:|
| R Random（解析值） | 0.000003315 | 0.000003315 | 0.000003315 | 0.000003315 |
| P Popularity | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L BM25 | 0.1653 | 0.0690 | 0.0706 | **0.1016** |
| S Frozen MiniLM | 0.0400 | 0.0036 | 0.0065 | **0.0167** |
| T NoGraph dual tower | 待导出 | 待导出 | 已完成训练 | 待定 |
| G GraphSAGE | 0.1355 | 0.1417 | 0.1508 | **0.1427** |

验证记录：

- sidecar 对 3,003,759 个快照模型逐行校验 id，然后为 12,680 个历史模型填缺失值；报告见
  [`SIDECAR_REPORT.json`](X6_runs/SIDECAR_REPORT.json)。
- 三个 seed 的候选池均为 3,016,439，eligible query 数为 1,476 / 1,101 / 1,545，model/dataset 映射相同；
  S 仅读取 `[64:448]`；G 逐 seed 精确复现 X4 数字。相应代码门均为 `true`。
- 旧 X6 沿用了 `#strictly-better + 1` 的并列排名，使全零或粗粒度 scorer 的大并列块全被记为第一名。
  修正为固定、标签无关的唯一次序后重跑了 P/L/S；上表是修正后结果。
- 最终代码上运行 `pytest scale1m/tests stage2TrainGraphSAGE/tests -q`：334 passed；X6 定向测试为 5 passed。
- T 的 resolved config 已与 X4-GD 逐键比对，唯一差异是 `num_layers: 1 -> 0`。seed 2 正式训练
  job `1529948` 完成 25 epoch，loss 24.2009 -> 17.9313，无 NaN，机制门通过；记录见
  [`X6NoGraph_full_s2_e25/`](X6_runs/X6NoGraph_full_s2_e25/)。seed 0/1 在 1 h 时限内完成到 epoch 19，
  已从 `last.pt` 提交自动续跑 job `1529988` / `1529989`，不会重算前 20 epoch。

调度调整：原 H200 任务预计两天后才开始，因此改用已通过 1-epoch smoke 的 L40S。L40S 在 1 h 内可完成
20--25 epoch，所以 seed 0/1 保留 checkpoint 后续跑；这只改变调度，不改变实验配方或优先级。

当前允许的最小结论只是：查询相关的 BM25 是最强的训练自由基线（0.1016），G 的三 seed 均值比它高
0.0410（约 1.40x）；结果不支持“每个 seed 都超过 BM25”。监督训练和图结构的独立增量必须等 T 的三 seed
`gold@10` 汇总后再按 §5 判定。

### 续跑结束后的命令

```powershell
# 1. 查看续跑状态（预计启动时间由 Slurm 动态决定）
ssh x98liu@watgpu.cs.uwaterloo.ca "squeue -j 1529988,1529989 -o '%.18i %.2t %.10M %.10l %R'; sacct -j 1529988,1529989 --format=JobID,State,ExitCode,Elapsed -n -P"

# 2. 任务完成后拉回 seed 0/1（seed 2 已在 X6_runs）
scp -r x98liu@watgpu.cs.uwaterloo.ca:/u801/x98liu/model_lake/runs/X6NoGraph_full_s0_e25 docs/1M/X6_runs/
scp -r x98liu@watgpu.cs.uwaterloo.ca:/u801/x98liu/model_lake/runs/X6NoGraph_full_s1_e25 docs/1M/X6_runs/

# 3. 复用现有 export_rf，只导出 T 的 held-out 嵌入
foreach ($s in 0,1,2) {
  .venv\Scripts\python.exe -m scale1m.export_rf `
    --run "docs/1M/X6_runs/X6NoGraph_full_s${s}_e25" --stage embed `
    --out "D:/research/model_lake/data/data1m/exports_x6/X6NoGraph_full_s${s}_e25"
}

# 4. 合并 T 与已完成的 P/L/S/G，并执行 config/mapping/query-count 门
.venv\Scripts\python.exe -m scale1m.eval_x6 --stage learned --seeds 0 1 2
Copy-Item D:/research/model_lake/data/data1m/metrics_x6/X6_BASELINES.json `
  docs/1M/X6_runs/X6_BASELINES.json -Force
```
