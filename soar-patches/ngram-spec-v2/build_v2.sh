#!/bin/bash
set -euo pipefail
D=/root/autodl-tmp/codex-drafts
V1=$D/submission-nvfp4-official4o6-gateup-preserve-scale2-v1
V2=$D/submission-nvfp4-official4o6-gateup-spec-ngram-v2
SPEC=$D/spec_engine/python

echo "=== [1] fresh copy v1 -> v2 ==="
rm -rf "$V2"
cp -a "$V1" "$V2"

echo "=== [2] swap engine: v2/sglang/python <- spec_engine/python ==="
rm -rf "$V2/sglang/python"
cp -a "$SPEC" "$V2/sglang/python"

echo "=== [3] clean dev cruft from v2/sglang ==="
find "$V2/sglang" -name "*.bak*" -delete 2>/dev/null || true
find "$V2/sglang" -name "*.orig" -delete 2>/dev/null || true
find "$V2/sglang" -name "*.pre_eagle3*" -delete 2>/dev/null || true
find "$V2/sglang" -name "*.pyc" -delete 2>/dev/null || true
find "$V2/sglang" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$V2/sglang" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo "=== [4] edit prepare_env.sh: ngram args + fix env vars + C++ warm-up ==="
python3 - <<PYEDIT
import re
p = "$V2/prepare_env.sh"
src = open(p).read()
lines = src.split("\n")
out = []
done_args = False
NGRAM = ("--speculative-algorithm NGRAM --speculative-num-steps 3 "
         "--speculative-num-draft-tokens 8 --speculative-ngram-branch-length 20 "
         "--speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1")
for ln in lines:
    if ln.startswith("export SGLANG_SERVER_ARGS="):
        m = re.match(r'export SGLANG_SERVER_ARGS="(.*)"\s*\$', ln)
        assert m, "SGLANG_SERVER_ARGS line not matched: %r" % ln
        base = m.group(1)
        out.append("# --- v2-ngram: speculative decoding consistency fixes (verify==decode) ---")
        out.append("export SPEC_NGRAM_FIX_GLA=1")
        out.append("export SPEC_NGRAM_VERIFY_MATCH_DECODE=1")
        out.append('export SGLANG_SERVER_ARGS="%s %s"' % (base, NGRAM))
        done_args = True
    else:
        out.append(ln)
assert done_args, "did not find SGLANG_SERVER_ARGS export"

# insert C++ ngram warm-up (fail-fast JIT build) before the final done echo
final = []
warm = '''
# --- v2-ngram: pre-build & verify the C++ ngram trie extension (needs g++ + ninja, c++20) ---
echo "[prepare_env] warming up ngram C++ extension (JIT compile)"
if ! "\${EVAL_PY}" - <<'PYNG'
from sglang.srt.speculative.cpp_ngram.ngram_cache import NgramCache
c = NgramCache(branch_length=20, draft_token_num=8)
c.batch_put([[1, 2, 3, 4, 5, 6, 7, 8]]); c.synchronize()
c.batch_get([[1, 2, 3]])
print("[prepare_env] ngram C++ extension OK")
PYNG
then
  soar_fail "[prepare_env] ngram C++ extension build/import failed (requires g++ with -std=c++20 and ninja)"
fi
'''
inserted_warm = False
for ln in out:
    if ln.startswith('echo "[prepare_env] done'):
        final.extend(warm.split("\n"))
        inserted_warm = True
    final.append(ln)
assert inserted_warm, "did not find final done echo"
open(p, "w").write("\n".join(final))
print("prepare_env.sh edited OK")
PYEDIT

echo "=== [5] edit run_local_test.sh: ngram args + cgbs 1 8 32 + fix env vars ==="
python3 - <<PYEDIT2
import re
p = "$V2/run_local_test.sh"
src = open(p).read()
lines = src.split("\n")
out = []
NGRAM = ("--speculative-algorithm NGRAM --speculative-num-steps 3 "
         "--speculative-num-draft-tokens 8 --speculative-ngram-branch-length 20 "
         "--speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1")
edited_args = False
added_env = False
for ln in lines:
    if ln.strip() == "export TRANSFORMERS_OFFLINE=1" and not added_env:
        out.append(ln)
        out.append("export SPEC_NGRAM_FIX_GLA=1")
        out.append("export SPEC_NGRAM_VERIFY_MATCH_DECODE=1")
        added_env = True
        continue
    if ln.startswith("SGLANG_ARGS="):
        m = re.match(r'SGLANG_ARGS="(.*)"\s*\$', ln)
        assert m, "SGLANG_ARGS not matched: %r" % ln
        base = m.group(1)
        base = base.replace("--cuda-graph-bs 1 8 32 64", "--cuda-graph-bs 1 8 32")
        out.append('SGLANG_ARGS="%s %s"' % (base, NGRAM))
        edited_args = True
        continue
    out.append(ln)
assert added_env, "did not add env exports"
assert edited_args, "did not edit SGLANG_ARGS"
open(p, "w").write("\n".join(out))
print("run_local_test.sh edited OK")
PYEDIT2

echo "=== [6] verify edits ==="
echo "--- prepare_env.sh: spec lines ---"
grep -nE "SPEC_NGRAM_FIX_GLA|SPEC_NGRAM_VERIFY_MATCH_DECODE|SGLANG_SERVER_ARGS=|ngram C\+\+|speculative-algorithm" "$V2/prepare_env.sh"
echo "--- run_local_test.sh: spec lines ---"
grep -nE "SPEC_NGRAM_FIX_GLA|SPEC_NGRAM_VERIFY_MATCH_DECODE|SGLANG_ARGS=" "$V2/run_local_test.sh"
echo "=== [7] confirm fixes present in v2 engine ==="
F="$V2/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
grep -c "SPEC_NGRAM_VERIFY_MATCH_DECODE\|is_current_stream_capturing" "$F"
grep -n "SPEC_NGRAM_FIX_GLA" "$V2/sglang/python/sglang/srt/speculative/ngram_worker.py"
echo "=== [8] v2 layout ==="
ls "$V2" | head -40
du -sh "$V2"
echo "=== BUILD_V2_DONE ==="
