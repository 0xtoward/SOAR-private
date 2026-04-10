#!/usr/bin/env bash
set -euo pipefail

ssh -p 50884 \
  -i /Users/ql/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@106.120.183.117 '
RUN_DIR=$(cat /root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_latest.txt)
echo RUN_DIR=$RUN_DIR
echo ---SUMMARY---
grep -E "Average Score|Total Duration|Overall TPS|Request failed|ERROR|HTTP|Traceback|Exception" \
  "$RUN_DIR/eval_model.log" "$RUN_DIR/server.log" || true
echo ---TAIL---
tail -n 80 "$RUN_DIR/eval_model.log" || true
'
