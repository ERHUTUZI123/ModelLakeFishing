set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
job=$(cat job_id_retry.txt)
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
tail -8 observations/train.log
tail -1 observations/resources.jsonl
if [ -f observations/RESOURCE_REPORT.json ]; then cat observations/RESOURCE_REPORT.json; fi
