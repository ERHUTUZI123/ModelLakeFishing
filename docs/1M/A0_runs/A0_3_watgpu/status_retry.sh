set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
job=$(cat job_id_retry.txt)
squeue -j "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
if [ -f "slurm-${job}.log" ]; then tail -12 "slurm-${job}.log"; fi
if [ -f boundary.log ]; then tail -10 boundary.log; fi
if [ -f observations/train.log ]; then tail -14 observations/train.log; fi
if [ -f observations/RESOURCE_REPORT.json ]; then cat observations/RESOURCE_REPORT.json; fi
