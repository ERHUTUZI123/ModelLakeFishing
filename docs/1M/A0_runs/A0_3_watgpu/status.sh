set -eu
cd /u801/x98liu/model_lake/a0_smoke_20260913
squeue -j 1539529
sacct -j 1539529 --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
scontrol show job 1539529
if [ -f slurm-1539529.log ]; then tail -20 slurm-1539529.log; fi
if [ -f boundary.log ]; then tail -8 boundary.log; fi
if [ -f observations/train.log ]; then tail -12 observations/train.log; fi
if [ -f observations/RESOURCE_REPORT.json ]; then cat observations/RESOURCE_REPORT.json; fi
