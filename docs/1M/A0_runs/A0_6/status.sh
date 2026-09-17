set -eu
cd /u801/x98liu/model_lake/a0_eval_20260914
job=$(cat job_id.txt)
squeue -j "$job" -o '%i %T %R'
squeue --start -j "$job"
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P
if [ -f "slurm-${job}.log" ]; then tail -5 "slurm-${job}.log"; fi
for stage in exact hnsw finalize; do
  if [ -f "${stage}.log" ]; then tail -6 "${stage}.log"; fi
done
