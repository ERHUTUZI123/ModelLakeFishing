"""Assemble A0.1 delivery from completed audit artifacts; no experiment execution."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "docs/1M/A0_runs"
AUDIT = OUT / "audit"
SOURCE_SHA = "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()

def read(rel): return json.loads((OUT/rel).read_text(encoding="utf-8"))
def write(rel,obj): (OUT/rel).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def bound(path): return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}

def main():
    if (AUDIT/"A0_DELIVERY_VALIDATION.json").exists():
        raise RuntimeError("Delivery already finalized; refuse to overwrite its audit baseline")
    inputs = read("audit/A0_INPUT_AUDIT.json")
    snapshots = read("audit/A0_SNAPSHOT_AUDIT.json")
    code = read("audit/A0_CODE_CAPTURE.json")
    cfg = read("audit/A0_CONFIG_AUDIT.json")
    old = read("A0_OLD_REFERENCE.json")
    inv = read("A0_METRIC_INVENTORY.json")
    proto = read("A0_PROTOCOL.json")
    # Required independent reviews must exist and be inspected before delivery.
    for name in ["A0_INPUT_REVIEW.json","A0_REFERENCE_REVIEW.json"]:
        if not (AUDIT/name).is_file(): raise FileNotFoundError(name)
        assert read("audit/"+name)["status"] == "PASS",name
    assert inputs["status"] == snapshots["status"] == "PASS"
    assert not inputs["errors"] and all(x["status"]=="PASS" for x in inputs["checks"])
    assert sha(REPO/"docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md") == SOURCE_SHA
    assert sha(OUT/"frozen/EVIDENCE_SOURCE_LIBRARY_en.md") == SOURCE_SHA
    assert sha(AUDIT/"audit_inputs.py") == inputs["script_sha256"]
    assert [s["eligible_queries"] for s in inputs["splits"]] == [1476,1101,1545]
    assert len({x["id"] for x in old["records"]}) == len(old["records"])
    assert len({x["id"] for x in inv["metrics"]}) == len(inv["metrics"])
    refs = {x["id"]:x for x in old["records"]}
    for row in inv["metrics"]:
        assert set(inv["required_fields"]) <= set(row),row["id"]
        assert row["new_value"] is None and row["status"]=="pending",row["id"]
        if row.get("old_reference_id"):
            ref = refs[row["old_reference_id"]]
            assert row["old_value"] == ref["value"],row["id"]
    changed_source = [x["path"] for x in code["implementation_files"] if sha(REPO/x["path"]) != x["sha256"]]
    assert not changed_source,changed_source
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO).decode().strip() == code["git_head"]
    before_status=(OUT/"frozen/git_status_before.txt").read_text(encoding="utf-8").splitlines()
    current_status=subprocess.check_output(["git","status","--porcelain=v1","--untracked-files=all"],cwd=REPO).decode("utf-8").splitlines()
    def external(lines):
        return sorted(x for x in lines if not x[3:].startswith("docs/1M/A0_runs/") and x[3:] not in {"docs/1M/A0.md","docs/1M/A0.1.md"})
    assert external(before_status)==external(current_status),"Unrelated workspace status changed"
    now=dt.datetime.now(dt.timezone.utc)
    local=now.astimezone(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S %Z")
    proto["status"]="A0.1_complete_protocol_and_frozen_inputs_verified_A0.2_not_started"
    proto["input_audit_binding"]={"status":"PASS","input_audit":bound(AUDIT/"A0_INPUT_AUDIT.json"),
        "snapshot_audit":bound(AUDIT/"A0_SNAPSHOT_AUDIT.json"),
        "input_review":bound(AUDIT/"A0_INPUT_REVIEW.json"),"query_identity":inputs["query_identity_artifact"]}
    proto["execution_boundary"]={"completed":"A0.1","next":"A0.2 pending separate execution",
        "training_started":False,"new_retrieval_results":False,"discrepancies":"A0_DISCREPANCIES.md"}
    for obj, name in [(proto,"A0_PROTOCOL.json"),(inv,"A0_METRIC_INVENTORY.json")]:
        obj["authorized_plan"]["snapshot_copy"]="frozen/A0.before_A01.md"
        obj["authorized_plan"]["hash_scope"]="Plan content at A0.1 entry, before feedback-only updates to live A0.md"
        write(name,obj)
    subprocess.run([sys.executable,"-B",str(AUDIT/"review_a01_reference.py")],cwd=REPO,check=True)
    assert read("audit/A0_REFERENCE_REVIEW.json")["status"]=="PASS"

    discrepancies="""# A0.1 差异与待处理事项

本表区分本步已查清的输入事实与 A0.2/A0.3 待完成的实现工作。事实口径唯一依据为冻结的英文事实文档。

| 编号 | 发现与证据 | 影响与处理 | 状态 |
|---|---|---|---|
| D01 | 原图 `x_dataset` 的 448–453、455 列含性能派生信息；独立按 `supervision_merged` 重算十列，逐元素完全相等。见 `audit/A0_INPUT_AUDIT.json`。 | 原图身份核验通过；泄漏修复仍待 A0.2 对全部节点七列置零，保留 454 和所有其他块。A0.1 没有修复泄漏。 | 已确认；A0.2 待执行 |
| D02 | 数据集快照有 `XSpaceCoderX/AD-Trajectories` / `XSpaceCoderX/AD-trajectories` 大小写碰撞，原始 ID 唯一数 1,008,417，规范化唯一数 1,008,416。分片 `hf_models_00001.jsonl.gz` 行 90,868 / 95,059。 | 两个 ID 及 basename 均未进入最终节点或卡片请求集合，影响最终节点 0。保留冻结输入/行表。初次保守 FAIL 与更正原因均保存在 snapshot audit 中。 | 已查清；不改变输入 |
| D03 | `build_config` 实际产生 40 个键，32 个有事实文档对应值，8 个只在当前实现中定义。见 `audit/A0_CONFIG_AUDIT.json`。 | 逐项冻结；未把实现独有字段伪称论文已报告。三个文档链接的旧 GD 配置仅用于交叉核对，40 键相同；不从 FXYZ 补指标或新规则。 | 已查清 |
| D04 | 现有评测含旧指标断言和效果阈值；A0 评测模式、七列图准备入口与 smoke-only 尚待实现。 | 按 A0 §6 实现必要适配并测试，保留计算/排序/候选池规则；新效果升降不能触发换方法。 | A0.2/A0.3 待执行 |
| D05 | exact 分块候选选择与最终稳定 tie-break 的一致性、split/export/prior 哈希绑定及训练恢复保护需要后续测试。 | 属于 A0.2/A0.3 的既定验收，A0.1 不宣称这些运行测试已通过。 | 待执行 |
| D06 | A0 中“每个 ef / 全部尝试值”需按唯一事实文档 §5.2 的“首个达标即选中”解释。 | 网格保持 [1000,1500,2000,3000,5000]，按原顺序测试到首个 recall≥0.99；逐个保存实际尝试值。之后未访问项显式标记 not_applicable，不能记 0、复制旧值或当作缺测失败。 | 已明确；未改原校准流程 |
| D07 | 快照、canonical、特征和图在本步通过文件身份与实际规模核验。 | 未重新抓取、重新运行六源归并、重新编码 MiniLM 或重建全量 KNN；这些是 A0 明确允许按哈希复用的冻结输入。本步不替代后续训练与结果复算。 | 范围已明确 |
| D08 | 初版 capture 脚本允许重复覆盖基线；输入审计的必需文件检查和关系名称比较可加强。 | 已保留实际执行过的脚本/首次输入报告；capture 新增已有基线即拒绝保护。输入脚本显式检查必需文件、按关系名称验证计数并更正来源节号，重新执行通过。 | 审计工具已修正 |

没有通过删除节点、改 query/gold、改 split、改分母、改变 K/prior 或使用旧检索产物来消除差异。
"""
    (OUT/"A0_DISCREPANCIES.md").write_text(discrepancies,encoding="utf-8")
    commands=f"""# A0.1 实际执行记录

工作目录：`{REPO}`。本步仅审计与文档回填，完成时间：{local}。

| 顺序 | 已执行动作 | 证据 |
|---|---|---|
| 1 | PowerShell 读取 A0 与唯一事实文档、检查路径/工作区；冻结文档、139 个实现文件、三个完整 Git diff 与原有未跟踪文件 | `audit/A0_CODE_CAPTURE.json`、`frozen/` |
| 2 | 对 72 个 gzip 分片完整流式计数、校验 SHA256/元数据绑定，追查大小写碰撞；修订报告保留初始诊断 | `audit/A0_SNAPSHOT_AUDIT.json` 的 execution 和 superseded_owned_audit |
| 3 | 提取唯一事实文档旧指标、冻结协议、登记全指标；纯函数解析配置并静态追查运行默认值 | 两个 build 脚本、`audit/A0_CONFIG_AUDIT.json` |
| 4 | 实际表/数组核验与三个 seed 的 split/query 身份重算。首次 337 项通过；加强审计存在性/关系名称检查后再运行 | `frozen/A0_INPUT_AUDIT.initial.json`、最终 `audit/A0_INPUT_AUDIT.json` |
| 5 | 独立核对 raw→canonical ID 顺序、lineage 与旧参照/清单 | `audit/A0_INPUT_REVIEW.json`、`audit/A0_REFERENCE_REVIEW.json` |
| 6 | 校验来源/代码不变、清单字段和旧新隔离，生成来源清单并回填 A0.1/A0 反馈 | `audit/finalize_a01.py`、`audit/A0_DELIVERY_VALIDATION.json` |

实际核心命令（PowerShell；路径参数详见各报告 execution/argv）：

```powershell
& '.venv/Scripts/python.exe' -B 'docs/1M/A0_runs/audit/capture_a01_state.py'
python 'docs/1M/A0_runs/audit/audit_snapshots.py' --data-root 'D:/research/model_lake/data' --docs-root 'D:/research/model_lake/codes/ModelLakeFishing/docs/1M' --output 'D:/research/model_lake/codes/ModelLakeFishing/docs/1M/A0_runs/audit/A0_SNAPSHOT_AUDIT.json'
& '.venv/Scripts/python.exe' -B 'docs/1M/A0_runs/audit/audit_inputs.py'
& '.venv/Scripts/python.exe' -B 'docs/1M/A0_runs/audit/finalize_a01.py'
```

首次 capture 使用的完整版本为 `frozen/capture_a01_state.executed.py`；现版本检测到基线后会拒绝再次覆盖。不要重复捕获来替换 A0.1 的起点。
snapshot 二次命令及修订理由已逐字保存在该报告中；该子审计使用系统 Python 3.13.1，输入审计使用仓库 `.venv` Python 3.13.1。
审计期间一次临时 `python -c` 表摘要读取因 PowerShell 引号解析失败，随后改用 here-string 标准输入读取成功；失败命令未写文件、未运行实验。

没有训练 job ID、epoch、checkpoint、export、sidecar 或 HNSW 产物；A0 的未来正式命令仍受各阶段前置条件约束。
审计脚本可用于复核，但现有结果应先保存为独立版本，不能覆盖原始冻结证据。
"""
    (OUT/"A0_COMMANDS.md").write_text(commands,encoding="utf-8")
    m=inputs["measurements"]
    seed_lines="\n".join(f"| {s['seed']} | {s['eligible_queries']:,} | {s['query_roots']:,} | {s['positive_label_counts']['test']:,} | {s['message_edge_counts']['test']:,} | 通过 |" for s in inputs["splits"])
    document=f"""# A0.1 执行与反馈

**状态：A0.1 已完成。** 更新时间：{local}。本步完成了协议、旧对照和冻结输入核验；A0.2–A0.7 尚未执行，新检索指标尚未产生。

唯一最终系统固定为 **X4G+D（GD 合并臂）→ HNSW top-1000 → task prior → top-10**。本步遵循 [A0.md](A0.md) §6 的 A0.1，仅以 [英文事实文档](EVIDENCE_SOURCE_LIBRARY_en.md) 确定历史事实与方法。F/X/Y/Z 中的链接只用于定位和交叉核对实物。

## 1. A0.1 五项执行结果

| A0.1 要求 | 已完成工作 | 证据 |
|---|---|---|
| 冻结事实文档与旧对照 | 文档逐字快照、SHA256；{len(old['records'])} 条旧参照保留来源节号、字面值和显示精度 | [冻结文档](A0_runs/frozen/EVIDENCE_SOURCE_LIBRARY_en.md)、[旧参照](A0_runs/A0_OLD_REFERENCE.json) |
| 保存代码状态 | Git HEAD、完整 staged/unstaged/HEAD 差异，139 个相关实现文件和 3 个原有未跟踪文件归档；当前实现重新核验无变化 | [代码捕获](A0_runs/audit/A0_CODE_CAPTURE.json) |
| 核验冻结输入 | 72 个快照分片完整读取；canonical、行表、词表、原图逐文件 SHA256 与实际计数；输入审计 {len(inputs['checks'])} 项通过 | [快照审计](A0_runs/audit/A0_SNAPSHOT_AUDIT.json)、[输入审计](A0_runs/audit/A0_INPUT_AUDIT.json) |
| 固定协议与允许变更 | 七列置零范围、三个 seed/25 epoch、GD 配置、K=1000、beta=1、shrink=5、排序和 ef 校准规则均登记 | [协议](A0_runs/A0_PROTOCOL.json)、[配置审计](A0_runs/audit/A0_CONFIG_AUDIT.json) |
| 全指标登记 | {len(inv['metrics'])} 个登记项均有定义、单位、scope、汇总方法、旧来源、未来重算输入与状态；新值全部 null/pending | [指标清单](A0_runs/A0_METRIC_INVENTORY.json) |

事实文档 SHA256：`{SOURCE_SHA}`。
Git HEAD：`{code['git_head']}`。起点工作区原本有本地改动，因此同时保存三份完整 binary diff；仅 HEAD 不能重现该状态。
冻结输入与每份交付文件的字节级绑定统一见 [来源清单](A0_runs/A0_SOURCE_MANIFEST.json)。

## 2. 从实际输入重新核对的结果

| 项目 | 本轮实际核验值 | 结论 |
|---|---:|---|
| 模型快照 | 61 个分片；3,003,759 条/唯一模型 ID | 与事实文档一致 |
| 数据集快照 | 11 个分片；1,008,417 条/原始唯一 ID | 与事实文档一致；大小写碰撞另列 |
| canonical 模型前缀 | 3,003,759 行；61 个 part | 与快照身份顺序及模型行表前缀对应 |
| 历史补充模型 | 12,680 行 | 规范化 ID 有序、无交叉、size 未知 |
| 全候选模型 | 3,016,439 行 | mappedID 连续，图行表与 ladder 一致 |
| dataset–task 节点 | 18,729 行；9,341 个 root | 节点/任务序列化、排序、metadata 与行表一致 |
| 正向性能边 | 247,803 条 | 六来源计数一致，pair 唯一，权重与图数组逐项对应 |
| 数据集相似边 / 正向 lineage | 374,580 / 859,065 | 原始相似边每源 20 条；反向关系对应 |
| 含反向关系的总边数 | 2,588,316 | 与事实文档一致 |
| 模型 / 数据集特征 | float32 [3,016,439,448] / [18,729,458] | 全数组有限，无全零行；文件身份匹配 |
| family 词表 | 41,056 行 | 哈希、ID 范围及图内词表一致 |
| unknown size / Other family | {m['unknown_size_percent']:.9f}% / {m['other_family_percent']:.9f}% | 按旧文档精度为 71.929% / 13.561% |
| 匹配 HF 卡片 | 3,928 / 18,729 | exact 3,621，parent 307；未匹配维持原标签 |
| 独立 gold_eligible 标志 | {m['gold_eligible_nodes']:,} 个节点 | 继续保留在评测表；不因特征清零改标签 |

原图实际文件逐个哈希后重建 digest：`{m['graph_digest']}`，与事实文档一致。不是仅相信 meta 中写下的 digest。
原图十列统计量已从冻结 supervision 与节点表独立重算，和 `x_dataset[:,448:458]` 逐元素完全一致。七个拟清零列仍有非零值，这确认了原始问题；**本步尚未修复泄漏，也没有训练可供报告的新模型。**

## 3. 三个 seed 的划分与查询身份

在只含性能边、节点数的临时内存数据对象上调用现有 split 函数，没有加载模型特征或调用训练。重新筛选候选并核验独立 gold_eligible；资格规则另写一次复核，两者 query ID 完全一致。

| seed | 合格查询数 | 查询 root 数 | test 正向性能边 | train+val 可见性能边 | test root 从 prior 来源排除 |
|---:|---:|---:|---:|---:|---|
{seed_lines}

训练、验证和测试 roots 两两不相交，各 split 的反向 message 性能边与正向精确对应。这里核验的是 prior **来源边集合**；新的 task-prior sidecar 留待 A0.5 重新聚合生成。
4,122 是三次划分下的查询观察总数。逐 query 的节点、root、历史候选数量及候选/权重 hash 已保存为 [查询身份清单](A0_runs/audit/A0_QUERY_IDENTITY.jsonl)，后续用于核对分母和身份。

## 4. 旧结果与新指标的边界

旧最终 gold@10 为 `0.3279132791 / 0.3387829246 / 0.2427184466`，三 seed 均值 `0.3031382168`；旧 p50/p95 为 `0.694 / 1.223 ms`。这些仅是英文事实文档中的历史对照。
全指标清单包含最终质量、原计分器额外分项、exact 参照、retention、coverage、每次实际 ef 尝试、延迟、索引/训练/导出成本及训练记录。事实文档未报告的旧值明确留空；未来必须从 A0 新产物重算适用项。
retention 按逐 seed 比值再取平均；延迟按逐 seed 分位数再取平均；条件 gold 位置中位数使用“gold 已进入候选池”的独立分母。exact 全库和 exact top-1000 只为同一 GD 的诊断参照。

## 5. 差异与复核范围

已查清一对数据集大小写碰撞，影响最终节点 0，保持冻结输入。协议已明确 ef 依原规则选首个达标项，记录实际尝试值；未访问项有独立状态。其余七列修复、smoke、A0 评测模式、tie/产物绑定测试仍由后续步骤负责。
详细逐项处理见 [差异清单](A0_runs/A0_DISCREPANCIES.md)。独立复核见 [输入复核](A0_runs/audit/A0_INPUT_REVIEW.json) 和 [参照复核](A0_runs/audit/A0_REFERENCE_REVIEW.json)。
本步验证冻结输入的身份和实际统计，没有重新采集 hub、重放全部六源数据归并、重编码 MiniLM 或重建全量 KNN；这些输入按 A0 的既定规则复用。通过本步不能替代 A0.2/A0.3 的实现验证。

## 6. 执行反馈区

| 项目 | 当前反馈 |
|---|---|
| 当前阶段 | A0.1 已完成；A0.2 待执行 |
| 本步检查结果 | 冻结快照/输入身份、配置对应、旧参照登记及交付校验通过 |
| 已修复性能泄漏 | 否；仍是原图，修复属于 A0.2 |
| 已开始正式训练 | 否；三个 seed 均 0/25，未启动 |
| 是否有新检索结果 | 否；全部新指标 null/pending |
| 当前未解决的 A0.1 输入身份冲突 | 无；已发现大小写碰撞查明为最终 0 行影响 |
| 下一步前置材料 | 本页、协议、来源清单、差异清单齐全 |
| 下一步执行内容 | 严格按 A0.2 实现七列修复和必要评测适配，随后 A0.3 验证 |
| 命令与环境 | [实际执行记录](A0_runs/A0_COMMANDS.md)、[交付校验](A0_runs/audit/A0_DELIVERY_VALIDATION.json) |
| 用户/复核者补充反馈 | 待填写；新增差异应追加记录，保留已有证据 |
"""
    (REPO/"docs/1M/A0.1.md").write_text(document,encoding="utf-8")
    plan_path=REPO/"docs/1M/A0.md"
    plan=plan_path.read_text(encoding="utf-8")
    changes={
        "状态：**执行计划，尚未修改实现、提交训练或生成新结果。**":"状态：**A0.1 已完成；A0.2–A0.7 待执行，尚未提交训练或生成新检索结果。**",
        "当前可开始的是 **A0.1**。":"A0.1 已完成，证据见 [A0.1.md](A0.1.md)；下一步为 **A0.2**，尚未开始。",
        "| 最后更新 | 2026-09-12，计划编写阶段 |":f"| 最后更新 | {local}，A0.1 完成 |",
        "| 当前阶段 | 计划已写入；A0.1–A0.7 尚未开始执行 |":"| 当前阶段 | A0.1 已完成；A0.2–A0.7 待执行 |",
        "| 本次已完成 | 以英文事实文档锁定最终系统、旧对照和全指标；写明执行命令、实现缺口和验收要求 |":"| 本次已完成 | 冻结协议、代码/输入/旧参照；实际核验与交叉复核通过，详见 [A0.1.md](A0.1.md) |",
        "| 当前可做的下一步 | A0.1：生成协议/来源/指标清单，核验冻结输入 |":"| 当前可做的下一步 | A0.2：按既定范围实现七列修复和必要评测适配 |",
        "| A0.1 冻结协议与旧对照 | 待执行 | — | — | — | 核验事实文档与输入 |":f"| A0.1 冻结协议与旧对照 | 已完成 | 2026-09-12 / {local} | [A0.1.md](A0.1.md)、[A0_runs](A0_runs/) | 快照与输入核验通过；旧参照/全指标登记齐全 | A0.2 待执行 |"
    }
    for before,after in changes.items():
        if plan.count(before)!=1: raise RuntimeError("Unexpected plan feedback state: "+before)
        plan=plan.replace(before,after)
    plan_path.write_text(plan,encoding="utf-8")
    versions={}
    for name in ["numpy","pandas","pyarrow","torch","torch-geometric","faiss-cpu","faiss-gpu","sentence-transformers"]:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]="not installed under this distribution name"
    validation={"stage":"A0.1","status":"PASS","checked_at_utc":now.isoformat(),
        "source_unchanged":True,"139_implementation_files_unchanged":True,"git_head_unchanged":True,
        "unrelated_workspace_status_unchanged":True,"old_reference_records":len(old["records"]),
        "metric_inventory_entries":len(inv["metrics"]),"new_values_all_null_pending":True,
        "inventory_required_fields_and_old_reference_bindings_valid":True,
        "input_audit_checks":len(inputs["checks"]),"input_snapshot_reports_pass":True,
        "runtime_distribution_versions":versions,"training_started":False,"new_retrieval_metrics_generated":False,
        "audit_revisions":{"initial_input_audit":"frozen/A0_INPUT_AUDIT.initial.json",
            "initial_script":"frozen/audit_inputs.initial.py","changes":"required-file guards; named relation count equality; correct section 5.3 reference","reexecuted":True},
        "script_sha256":sha(Path(__file__))}
    write("audit/A0_DELIVERY_VALIDATION.json",validation)
    artifacts=[bound(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="A0_SOURCE_MANIFEST.json"]
    manifest={"schema":"a0.source_manifest.v1","stage":"A0.1","status":"complete",
        "created_at_utc":now.isoformat(),"authority":bound(REPO/"docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md"),
        "authority_snapshot":bound(OUT/"frozen/EVIDENCE_SOURCE_LIBRARY_en.md"),
        "plan_at_entry":bound(OUT/"frozen/A0.before_A01.md"),"plan_with_feedback":bound(plan_path),
        "delivery":bound(REPO/"docs/1M/A0.1.md"),"git_head":code["git_head"],
        "inputs":{"snapshot_audit":"audit/A0_SNAPSHOT_AUDIT.json","input_audit":"audit/A0_INPUT_AUDIT.json",
            "identity_review":"audit/A0_INPUT_REVIEW.json","binding_policy":"Nested audits bind original files with full hashes/counts; original inputs retained in place."},
        "artifacts":artifacts,"self_hash_policy":"Manifest excludes itself; all listed artifacts carry content hashes.",
        "later_stages":"No new graph, model, embeddings, priors, index, or retrieval result yet; append stage-specific provenance when executed."}
    write("A0_SOURCE_MANIFEST.json",manifest)
    for item in manifest["artifacts"]:
        assert sha(Path(item["path"]))==item["sha256"]
    print(json.dumps({"A0.1":"complete","artifacts":len(artifacts),"input_checks":len(inputs["checks"]),
        "old_reference_records":len(old["records"]),"metric_inventory_entries":len(inv["metrics"]),"new_retrieval_results":False}))

if __name__=="__main__": main()
