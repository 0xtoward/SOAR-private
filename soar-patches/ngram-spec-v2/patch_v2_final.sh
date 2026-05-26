#!/bin/bash
set -euo pipefail
V2=/root/autodl-tmp/codex-drafts/submission-nvfp4-official4o6-gateup-spec-ngram-v2
for f in "$V2/prepare_env.sh" "$V2/run_local_test.sh"; do
  python3 - "$f" <<'PYEDIT'
import sys
p = sys.argv[1]
s = open(p).read()
# 1) draft 8 -> 12
assert "--speculative-num-draft-tokens 8" in s, "draft=8 anchor missing in %s" % p
s = s.replace("--speculative-num-draft-tokens 8", "--speculative-num-draft-tokens 12")
# 2) add SPEC_NGRAM_BATCH_GATE=8 right after the MATCH_DECODE export (idempotent)
anchor = "export SPEC_NGRAM_VERIFY_MATCH_DECODE=1"
assert anchor in s, "match-decode export missing in %s" % p
if "SPEC_NGRAM_BATCH_GATE" not in s:
    s = s.replace(anchor, anchor + "\nexport SPEC_NGRAM_BATCH_GATE=8  # bs>8 -> plain decode (bs=32 spec-verify cudagraph crashes); spec only at bs<=8", 1)
# 3) ensure NO max-running-requests cap (gate handles high bs)
assert "--max-running-requests" not in s, "unexpected max-running-requests in %s" % p
open(p, "w").write(s)
print("patched", p)
PYEDIT
done
echo "=== resulting spec env + server-arg lines ==="
grep -nE "SPEC_NGRAM_FIX_GLA|SPEC_NGRAM_VERIFY_MATCH_DECODE|SPEC_NGRAM_BATCH_GATE|SGLANG_SERVER_ARGS=|^SGLANG_ARGS=" "$V2/prepare_env.sh" "$V2/run_local_test.sh"
bash -n "$V2/prepare_env.sh" && echo "prepare_env.sh SYNTAX OK"
bash -n "$V2/run_local_test.sh" && echo "run_local_test.sh SYNTAX OK"
echo PATCH_FINAL_DONE
