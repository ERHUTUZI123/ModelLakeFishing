# A0.1 实际执行记录

工作目录：`D:\research\model_lake\codes\ModelLakeFishing`。本步仅审计与文档回填，完成时间：2026-09-12 20:22:14 EDT。

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


## A0.2 执行记录

本段追加于 A0.2，A0.1 原文保留；追加前版本在 `A0_2/frozen/A0_COMMANDS.md`。
工作目录固定为 `D:\research\model_lake\codes\ModelLakeFishing`，解释器固定为仓库 `.venv\Scripts\python.exe`。

实际已执行的新图命令如下；**这是执行记录，不要对已有输出重复运行**：

```powershell
& '.venv/Scripts/python.exe' -B -m scale1m.prepare_a0_graph `
  --source 'D:\research\model_lake\data\data1m\graphs\hgraph_rf' `
  --out 'D:\research\model_lake\data\data1m\a0_20260912\graph'
```

首版通过七列/文件核验后，独立审查发现关键编码元数据未进入 digest 绑定。已在核验同一 A0 根目录的绝对路径后，用 PowerShell `Move-Item -LiteralPath` 保存为 `graph_before_metadata_binding` 和 `graph_before_metadata_binding.verification.json`；没有删除或修改旧输入。修正元数据绑定后再次执行上述准备命令，退出码 0，耗时 27.3801943 秒，最终 digest 为 `acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db`。两版均未用于训练。

实际执行了以下检查包装器，记录精确 argv、日志、退出码、源码 SHA 和耗时。联合检查只列出 A0.2 相关小型 fixture，没有执行 A0.3 全套测试或真实 smoke：

```powershell
& '.venv/Scripts/python.exe' -B 'docs/1M/A0_runs/A0_2/check_a02_interfaces.py'
& '.venv/Scripts/python.exe' -B 'docs/1M/A0_runs/A0_2/run_a02_focused_checks.py' --label final
```

最终结果：**189 passed，0 failed/error/skipped**，pytest 17.83 秒，包装器 19.469 秒；7 个 CLI 检查通过，40 键配置不变。

证据：`A0_2/A0_INTERFACE_CHECKS.json`、`A0_2/final_focused_checks.json`、`A0_2/final_focused_tests.log`、`A0_2/final_focused_tests.xml`。包装器初版一次因 repo 路径层级错误在 import 前退出；修正后执行。之前的 74 项联合测试、15 项元数据测试，以及独立复核的初始失败和修正证据均保留；它们不替代最终源码上的联合检查。

独立新图审查见 `A0_2/A0_GRAPH_REVIEW.json`；五个历史源的实际文件身份核验见 `A0_2/A0_HISTORICAL_SOURCE_IDENTITY.json`。旧报告只用于来源身份比对，未把其数字填入本轮新测量。

### 后续 A0.7：来源重算与独立复算命令（本步未执行）

以下命令在 A0.6 真实 exact/HNSW/finalize 完成后执行；本段使 A0.2 实现入口可交接，不提前运行后续实验。

```powershell
$A0Repo = 'D:\research\model_lake\codes\ModelLakeFishing'
$A0Python = Join-Path $A0Repo '.venv\Scripts\python.exe'
$A0Protocol = Join-Path $A0Repo 'docs\1M\A0_runs\A0_PROTOCOL.json'
$A0Inventory = Join-Path $A0Repo 'docs\1M\A0_runs\A0_METRIC_INVENTORY.json'
$A0Metrics = 'D:\research\model_lake\data\data1m\a0_20260912\metrics'
$A0Independent = Join-Path $A0Metrics 'independent'
$A0SourceCounts = Join-Path $A0Independent 'A0_SOURCE_COUNTS.json'
$A0RecordSources = Join-Path $A0Independent 'A0_RECORD_SOURCES.json'
Set-Location -LiteralPath $A0Repo

& $A0Python -B -m scale1m.a0_source_counts `
  --protocol $A0Protocol --out $A0SourceCounts `
  --historical-dir (Join-Path $A0Repo 'stage1BuildTransferGraph') `
  --historical-bindings (Join-Path $A0Repo 'docs\1M\A0_runs\A0_2\A0_HISTORICAL_SOURCE_IDENTITY.json')
if ($LASTEXITCODE -eq 1) { throw 'A0 source identity or recount failed' }
# 退出码 2 表示仍有原始事件证据缺口。可继续生成缺项报告，不能认定完整验收通过。

& $A0Python -B -m scale1m.recompute_a0 `
  --raw $A0Metrics --protocol $A0Protocol --inventory $A0Inventory `
  --out $A0Independent --make-record-template $A0RecordSources
if ($LASTEXITCODE -ne 0) { throw 'A0 record-source descriptor generation failed' }

& $A0Python -B -m scale1m.recompute_a0 `
  --raw $A0Metrics --protocol $A0Protocol --inventory $A0Inventory `
  --out $A0Independent --records $A0RecordSources --source-counts $A0SourceCounts
if ($LASTEXITCODE -eq 1) { throw 'A0 independent consistency verification failed' }
if ($LASTEXITCODE -eq 2) { throw 'A0 report generated with missing evidence; full completion remains unaccepted' }
```

独立输出为 `A0_REPORT.json`、`A0_RESULTS.md` 和新的 `A0_METRIC_INVENTORY.json`；初始 A0.1 库存不被覆盖。退出码 0 才表示完整；1 表示完整性/协议/复算不一致；2 表示真实缺测。两个快照事件计数的原始证据尚未补齐时，预期仍为 2，不得以旧 PROVENANCE 数字使其通过。

文档回填和归档实际使用 `A0_2/update_a02_feedback.py`、`A0_2/capture_a02_delivery.py`。终检修正了 PowerShell 管道对中文的错误编码，之后由 UTF-8 文件直接追加；失败版本保留在 `A0_2/frozen/*encoding_failure.md`，未修改实验代码、原图或 A0.1 冻结合同。
