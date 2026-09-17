set -eu
cd /u801/x98liu/model_lake/a0_export_20260913
job=$(cat job_id.txt)
squeue -j "$job" -o '%i %T %R'
squeue --start -j "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
if [ -f "slurm-${job}.log" ]; then tail -6 "slurm-${job}.log"; fi
for seed in 0 1 2; do
  if [ -f "validation_s${seed}.log" ]; then tail -5 "validation_s${seed}.log";
  elif [ -f "prior_s${seed}.log" ]; then tail -3 "prior_s${seed}.log";
  elif [ -f "export_s${seed}.log" ]; then tail -3 "export_s${seed}.log"; fi
done
