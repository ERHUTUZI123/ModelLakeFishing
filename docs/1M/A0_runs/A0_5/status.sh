set -eu
cd /u801/x98liu/model_lake/a0_export_20260913
squeue -j 1540848 -o '%i %T %R'
sacct -j 1540848 --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
if [ -f slurm-1540848.log ]; then tail -7 slurm-1540848.log; fi
for seed in 0 1 2; do
  if [ -f "validation_s${seed}.log" ]; then tail -5 "validation_s${seed}.log";
  elif [ -f "prior_s${seed}.log" ]; then tail -3 "prior_s${seed}.log";
  elif [ -f "export_s${seed}.log" ]; then tail -3 "export_s${seed}.log"; fi
done
