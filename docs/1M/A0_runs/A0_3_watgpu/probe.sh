set -eu
source ~/.mlf_env
printf 'PATHS\n%s\n%s\n%s\n%s\n' "$PROJECT_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT" "$ENV_SOURCE"
ls -ld "$PROJECT_ROOT" "$DATA_ROOT/data1m/graphs/hgraph_rf" "$DATA_ROOT/data1m/a0_20260912/graph" 2>/dev/null || true
find "$HOME" -maxdepth 2 -name '*runtime*.env' -print
ls "$OUTPUT_ROOT/envs"
scontrol show node watgpu308
scontrol show node watgpu608
sinfo -o '%P %a %l %D %G'
du -sh "$ENV_SOURCE"
command -v python3
find "$PROJECT_ROOT" -name AGENTS.md -not -path '*/.git/*' -not -path '*/.venv/*' -print
