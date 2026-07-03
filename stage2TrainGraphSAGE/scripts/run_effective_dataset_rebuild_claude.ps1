$ErrorActionPreference = "Stop"

$workspace = "D:\research\model_lake\codes\ModelLakeFishing"
$guide = Join-Path $workspace "stage2TrainGraphSAGE\docs\EFFECTIVE_DATASET_RECOLLECTION_AND_AB_GUIDE.md"
$artifactDir = Join-Path $workspace "stage2TrainGraphSAGE\artifacts\scheduled_claude"
$claude = "C:\Users\17982\AppData\Roaming\npm\claude.ps1"

New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$startedAt = Get-Date
$stamp = $startedAt.ToString("yyyyMMdd_HHmmss")
$logPath = Join-Path $artifactDir "effective_dataset_rebuild_$stamp.log"
$statusPath = Join-Path $artifactDir "effective_dataset_rebuild_$stamp.status.json"

$prompt = @"
Open and execute the complete instruction document at:
$guide

This is an implementation and experiment task, not a request to summarize the
document. Work autonomously from Phase 0 onward, use the existing repository as
evidence, preserve all current artifacts, and create the isolated v2 outputs
required by the guide. Run the required tests and audits. Do not fabricate
performance records or silently change the frozen 2,000-model universe. If a
long external collection phase cannot finish in one session, leave resumable
artifacts, exact commands, and a precise progress/blocker report in the locations
specified by the guide. Continue through Stage-1 reconstruction and Stage-2 A/B
testing as far as the verified data permits.
"@

if (-not (Test-Path -LiteralPath $guide)) {
    throw "Instruction guide not found: $guide"
}
if (-not (Test-Path -LiteralPath $claude)) {
    throw "Claude CLI not found: $claude"
}

Push-Location $workspace
try {
    & $claude `
        --print `
        --model "claude-opus-4-8" `
        --effort "high" `
        --permission-mode "auto" `
        --output-format "text" `
        --name "effective-dataset-rebuild-v2" `
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

$status = [ordered]@{
    started_at = $startedAt.ToString("o")
    finished_at = (Get-Date).ToString("o")
    exit_code = $exitCode
    model = "claude-opus-4-8"
    effort = "high"
    permission_mode = "auto"
    guide = $guide
    log = $logPath
}
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8

exit $exitCode
