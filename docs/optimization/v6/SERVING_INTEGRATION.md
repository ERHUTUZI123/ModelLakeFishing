# v6 serving 层集成可审计记录 —— 统一 sibling+task 重排接入 Stage-3

**日期:** 2026-07-19
**上游:** v6 §0.3/§0.7/§0.9(S2 采纳、S1 否决、P2b 采纳)
**性质:** 把两个已采纳的零重训 serving 融合(S2 sibling + P2b task)接入已上线
的 `d0_L1L3b` 服务栈——冻结的 z_m/z_d/HNSW 索引**一律不改**,只在检索后加一步。

---

## 1. 统一重排(无分支覆盖两类根)

```
fused(D, m) = minmax(z_d[D]·z_m)[m] + α·sibling_boost(m) + β·task_boost(m)   (α=β=1)
sibling_boost(m) = mean norm_acc(m, D')  over D' ∈ 湖内同根兄弟, D'≠D
task_boost(m)    = shrunk mean norm_acc(m, D') over D' ∈ 湖内同任务, D'≠D  (k=5)
```

**E19 分叉自动闭合、无需按根分支:** 无兄弟的查询 sibling_boost≡0(S2 静默),
无同任务的查询 task_boost≡0——两个 boost 各自在不适用处归零,一套逻辑同时
服务 sibling-rich(82%,S2 主导)与孤立根(18%,P2b 兑现)。

## 2. 代码改动(Stage-3,增量,不改已有路径)

| 文件 | 性质 |
|---|---|
| `stage3HNSW/build_prior_sidecar.py` | **新增**:从图导出 prior sidecar(trained_on 邻接 model/dataset/acc + root_id + task_id),mappedID 行序对齐 export 的 id 快照(铁律断言) |
| `stage3HNSW/serving_rerank.py` | **新增**:`load_serving`(读 export + sidecar);`fused_rerank(D, α, β, k)`——MIPS 池 + sibling/task boost 重排;`_prior` 显式排除 D 自身 |

sidecar(`prior_sidecar.npz`,54,795 边 + root/task 映射)与 export 同目录;
z_m/z_d/index.bin **零改动**(重排是检索后纯计算)。

## 3. 端到端验证(已上线 d0_L1L3b 索引)

| 查询类型 | has_sibling | has_task | 重排 top-10 变化 |
|---|---|---|---|
| sibling-rich(`amazon_counterfactual/de`) | True | True | ✓(两通道都生效) |
| 孤立(`abusive-clauses-pl/default`) | **False** | True | ✓(**S2 静默,P2b 独撑**——E19 设计) |

**合法性 + 冻结不变量(断言全过):**
1. 孤立查询 sibling_boost **逐元素为零**(无兄弟,S2 正确不触发);
2. **"仅与 D 有边"的模型 sibling_boost = 0**(D 自身被 `_prior` 的 `dd==d` skip
   排除)——绝不使用 D 的 held-out 标签,§0.6-3 合法性边界在 serving 代码里
   落实;
3. HNSW 索引文件 `index.bin` 全程未写(重排是检索后步骤)。

## 4. 复现

```bash
cd ModelLakeFishing  # 仓库根
# 1) 建 sidecar(一次,与 export 同目录)
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.build_prior_sidecar \
    --graph ModelLakeFishing/stage1BuildTransferGraph/hgraph_d0_v1.pt \
    --export ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b
# 2) 融合重排查询
ModelLakeFishing/.venv/Scripts/python.exe -m ModelLakeFishing.stage3HNSW.serving_rerank \
    --export ModelLakeFishing/stage3HNSW/artifacts/exports/d0_L1L3b \
    --dataset amazon_counterfactual/de --alpha 1 --beta 1
```

## 5. 状态与剩余

- **serving 层已具备统一 sibling+task 重排能力**,在已上线索引上验证通过,
  合法性/冻结不变量落实;offline 8-seed 已证 gold@10 增益(S2:sibling-rich
  0.162→0.266;P2b:孤立 0.121→0.180;叠加 all 0.137→0.219)。
- **不需要 Stage-3 重索引**:z_m/z_d/index 未变,只多了一个 sidecar +
  一步重排;晋升配置 L1L3b 及其索引保持不变。
- 剩余(v6 收尾):serving_rerank 的 α/β 可暴露为查询参数;retrieval_service
  接线(供 Stage-4 reranker);S3(条件行,须避 S1 两坑)。
- 边界:重排增益的绝对口径来自 node-level 服务场景(已知根的新兄弟 / 同任务
  查询);全新孤立根(无兄弟无同任务)两 boost 均零、回退纯 MIPS——诚实无害。