#!/usr/bin/env bash
# Main LOSO study on ADABase. Main task: n-back binary (enroll 3 seeds; strict/calib seed 0);
# secondary tasks (n-back 3-class, k-drive 3-class / binary): enroll protocol, seed 0.
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
N=${NSHARDS:-8}
run() {  # task protocol seeds...
  local T=$1 P=$2; shift 2
  echo "[$(date +%T)] $T $P seeds=$*"
  for i in $(seq 0 $((N-1))); do
    $PY -u scripts/a06_train.py --task $T --protocol $P --seeds "$@" --shard $i --nshards $N > logs/train_${T}_${P}_$i.log 2>&1 &
  done
  wait
}
run nback_bin enroll 0 1 2
run nback_bin strict 0
run nback_bin calib 0
for T in nback_3 kdrive_3 kdrive_bin; do run $T enroll 0; done   # secondary tasks: main protocol
echo "[$(date +%T)] done"; touch logs/MAIN_DONE
