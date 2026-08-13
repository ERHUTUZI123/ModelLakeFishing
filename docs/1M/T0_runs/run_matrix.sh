#!/usr/bin/env bash
# T0.3 run matrix -- resumable. Any run whose export_report.json already exists
# is skipped, so a killed session costs at most one run.
#
#   baseline_off : every T0 switch OFF  -> the pre-change anchor (G-A1 control)
#                  seed 0 doubles as the equivalence check against the untouched
#                  P3 artifact ml_sub_L1L3b (gold@10 = 0.4159)
#   allon_fullneg: items 1,2,4,5,7 ON, contrastive still full-N^2  -> isolates
#                  everything EXCEPT the objective change
#   allon_n256   : + sampled negatives n_neg=256                   -> D-14 row
#   allon_n1024  : + sampled negatives n_neg=1024                  -> D-14 row
#                  (only run when asked: pass ROWS="allon_n1024")
set -u
cd "$(dirname "$0")/../../../.."          # -> codes/
PY=ModelLakeFishing/.venv/Scripts/python.exe
GRAPH=ModelLakeFishing/stage1BuildTransferGraph/hgraph_ml_v2_sub.pt
EXPORTS=ModelLakeFishing/docs/scale/P3/exports
LOGS=ModelLakeFishing/docs/1M/T0_runs
mkdir -p "$LOGS"

ON="--fanout --sparse-M --chunked-infer 4000 --stream-scores --hnsw-threads 8 --iso-recall 0.99 --expect-n 12000"

ROWS="${ROWS:-baseline_off allon_fullneg allon_n256}"
SEEDS="${SEEDS:-0 1 2}"

for row in $ROWS; do
  case "$row" in
    baseline_off)  FLAGS="" ;;
    allon_fullneg) FLAGS="$ON" ;;
    allon_n256)    FLAGS="$ON --contrast-n-neg 256" ;;
    allon_n1024)   FLAGS="$ON --contrast-n-neg 1024" ;;
    *) echo "unknown row $row"; exit 2 ;;
  esac
  for s in $SEEDS; do
    tag="T0_${row}_s${s}"
    if [ -f "$EXPORTS/$tag/export_report.json" ]; then
      echo "SKIP  $tag (already complete)"
      continue
    fi
    echo "START $tag  $(date +%H:%M:%S)"
    $PY -m ModelLakeFishing.scale.export_ours --graph "$GRAPH" \
        --seed "$s" --epochs 25 --tag "$tag" $FLAGS \
        > "$LOGS/$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "FAILED $tag rc=$rc -- see $LOGS/$tag.log"
      tail -n 25 "$LOGS/$tag.log"
    else
      echo "DONE  $tag  $(date +%H:%M:%S)  $($PY -c "import json;d=json.load(open(r'$EXPORTS/$tag/export_report.json'));print('gold@10=%.4f train_sec=%.0f'%(d['our_gold_at_10'],d['train_sec']))")"
    fi
  done
done
echo "MATRIX COMPLETE"
