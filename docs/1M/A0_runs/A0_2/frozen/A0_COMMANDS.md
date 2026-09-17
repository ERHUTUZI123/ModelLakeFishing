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
