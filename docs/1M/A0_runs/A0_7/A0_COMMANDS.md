# A0.7 实际命令与环境

执行主机：本机 i7-14650HX；Python 3.13.1；CPU 8 线程。训练和正式检索的原始耗时/环境取自 A0.4–A0.6 记录。

9 个完整导出/索引文件通过 scp 下载，原始 SHA 全部核验；实际列表见 DOWNLOAD_PLAN.json。

以下为实际 argv（JSON 数组，保留精确参数和 Windows 路径）：

## source_counts

```json
{
  "command": [
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\.venv\\Scripts\\python.exe",
    "-X",
    "utf8",
    "-u",
    "-B",
    "-m",
    "scale1m.a0_source_counts",
    "--protocol",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_6\\inputs\\A0_PROTOCOL.linux.json",
    "--historical-bindings",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_2\\A0_HISTORICAL_SOURCE_IDENTITY.json",
    "--out",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_SOURCE_COUNTS.json",
    "--run-id",
    "A0_20260912"
  ],
  "pid": 24992,
  "started_at_utc": "2026-09-14T03:05:35.156774+00:00",
  "start_monotonic": 16259.9073085,
  "exit_code": 2,
  "elapsed_seconds": 431.4029724999982,
  "elapsed_scope": "Supervisor start-to-join interval, overlapping artifact downloads; not isolated source-recount CPU duration."
}
```

## record_template

```json
{
  "command": [
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\.venv\\Scripts\\python.exe",
    "-X",
    "utf8",
    "-u",
    "-B",
    "-m",
    "scale1m.recompute_a0",
    "--raw",
    "D:\\research\\model_lake\\data\\data1m\\a0_20260912\\metrics",
    "--protocol",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_6\\inputs\\A0_PROTOCOL.linux.json",
    "--inventory",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\frozen\\A0_METRIC_INVENTORY.json",
    "--out",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results",
    "--make-record-template",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_RECORD_SOURCES.json"
  ],
  "pid": 27172,
  "started_at_utc": "2026-09-14T03:12:46.565390+00:00",
  "start_monotonic": 16691.3159217,
  "exit_code": 0,
  "elapsed_seconds": 40.0156530000022
}
```

## recompute

```json
{
  "command": [
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\.venv\\Scripts\\python.exe",
    "-X",
    "utf8",
    "-u",
    "-B",
    "-m",
    "scale1m.recompute_a0",
    "--raw",
    "D:\\research\\model_lake\\data\\data1m\\a0_20260912\\metrics",
    "--protocol",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_6\\inputs\\A0_PROTOCOL.linux.json",
    "--inventory",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\frozen\\A0_METRIC_INVENTORY.json",
    "--out",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results",
    "--records",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_RECORD_SOURCES.json",
    "--source-counts",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_SOURCE_COUNTS.json"
  ],
  "pid": 21676,
  "started_at_utc": "2026-09-14T03:13:26.587361+00:00",
  "start_monotonic": 16731.337892,
  "exit_code": null,
  "completion_evidence": "CLI log reports incomplete and all report/inventory outputs exist; supplemental checks PASS. Original supervisor ended during turn interruption, so OS exit code and supervision endpoint were not captured.",
  "expected_cli_exit_from_completed_report": 2
}
```

源统计与文件传输并行；source_counts 的监督区间包含等待传输完成的时间，不能当作独占运行耗时。
用户中途转向更新 evidence library 时原监督进程终止，独立复算已写出完整报告；随后以 finalize_delivery.py 完成 470 项额外对照。操作系统退出码未采得，按未观测保留。
复算程序的预期退出码 2 表示仍有缺失项；本轮确认为两项采集事件日志。
finalize_delivery.py、backfill.py 在本阶段完成后执行；library_revision/preserve_evidence.py 为后续正文更新保存原文与可复现目录。
## A0.7 report unit correction

{
  "command": [
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\.venv\\Scripts\\python.exe",
    "-X",
    "utf8",
    "-u",
    "-B",
    "-m",
    "scale1m.recompute_a0",
    "--raw",
    "D:/research/model_lake/data/data1m/a0_20260912/metrics",
    "--protocol",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_6\\inputs\\A0_PROTOCOL.linux.json",
    "--inventory",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\frozen\\A0_METRIC_INVENTORY.json",
    "--out",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results",
    "--records",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_RECORD_SOURCES.json",
    "--source-counts",
    "D:\\research\\model_lake\\codes\\ModelLakeFishing\\docs\\1M\\A0_runs\\A0_7\\results\\A0_SOURCE_COUNTS.json"
  ],
  "exit_code": 2,
  "start_ns": 2342239894400,
  "end_ns": 2434329169700,
  "seconds": 92.0892753,
  "started_at_utc": "2026-09-15T01:43:47.489692+00:00",
  "reader_sha256": "3ce80ac6f066716bef0f8a7731ad6abcb25fb94d3f79f7d9cf9ce5b36cf0c0f9",
  "scope": "Independent report re-read and unit-contract repair; formal training, embeddings, raw query outputs and measured costs reused by verified hash.",
  "raw_measurement_values_and_statuses_unchanged": true,
  "native_quality_rows_identical": true,
  "measurements_compared": 982
}

