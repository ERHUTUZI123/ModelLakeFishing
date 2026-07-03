$ErrorActionPreference = "Stop"

$workspace = "D:\research\model_lake\codes\ModelLakeFishing"
$guide = Join-Path $workspace "stage2TrainGraphSAGE\docs\V2_FULL_VERSION_WARM_COLD_RERUN_GUIDE.md"
$logDir = Join-Path $workspace "stage2TrainGraphSAGE\artifacts\effective_dataset_v2\full_version_rerun\claude_logs"
$claude = "C:\Users\17982\AppData\Roaming\npm\claude.ps1"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$startedAt = Get-Date
$stamp = $startedAt.ToString("yyyyMMdd_HHmmss")
$logPath = Join-Path $logDir "opus48_full_rerun_$stamp.log"
$statusPath = Join-Path $logDir "opus48_full_rerun_$stamp.status.json"

$prompt = @"
Read and execute the complete instruction document at:
$guide

This is an implementation and experiment task, not a summary request. Build the
resumable v2 full-version driver, verify one warm and one cold B0 artifact, and
then run every listed version for the required warm and five-fold cold tables.
Use the existing configuration manifest as the source of truth. Persist each run
immediately, resume completed work, and append only completed final tables to the
specified review_everything.md. Do not substitute the earlier smoke-test values.
Continue autonomously as far as compute permits and leave an exact resumable
status if the sweep outlives this session.
"@

if (-not (Test-Path -LiteralPath $guide)) { throw "Guide not found: $guide" }
if (-not (Test-Path -LiteralPath $claude)) { throw "Claude CLI not found: $claude" }

Push-Location $workspace
try {
    & $claude `
        --print `
        --model "claude-opus-4-8" `
        --effort "high" `
        --permission-mode "auto" `
        --output-format "text" `
        --name "v2-full-version-warm-cold-rerun" `
        $prompt 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Tee-Object -FilePath $logPath -Append
    $exitCode = 1
}
finally {
    Pop-Location
}

[ordered]@{
    started_at = $startedAt.ToString("o")
    finished_at = (Get-Date).ToString("o")
    exit_code = $exitCode
    model = "claude-opus-4-8"
    effort = "high"
    guide = $guide
    log = $logPath
} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

exit $exitCode
