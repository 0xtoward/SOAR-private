#!/usr/bin/env python3
# FIX: NGRAM worker never advanced the GLA recurrent state after verify (eager/cudagraph
# both corrupted -> garbage output). EAGLE worker does this via
# update_simple_gla_state_after_target_verify; NGRAM (ngram_worker.py) was missing it.
# accepted_steps == verify_input.accept_length (ngram's accept_length == eagle's
# accept_length_per_req_cpu; eagle uses accepted_steps = (that+1)-1 = that).
# Opt-in via SPEC_NGRAM_FIX_GLA=1. Backs up to .bak_fixgla, py_compile-verifies.
import os, shutil, py_compile, sys

F = "/root/autodl-tmp/codex-drafts/spec_engine/python/sglang/srt/speculative/ngram_worker.py"
src = open(F).read()
bak = F + ".bak_fixgla"
if not os.path.exists(bak):
    shutil.copy(F, bak); print("backup ->", bak)
else:
    print("backup exists ->", bak)

changes = 0

# 1) __init__ flag (after the prompt-index set, which the prior patch added)
anchor1 = "        self._prompt_indexed_rids = set()\n"
if "self._fix_gla_state" not in src:
    assert anchor1 in src, "anchor1 (_prompt_indexed_rids) not found"
    src = src.replace(anchor1, anchor1 + '        self._fix_gla_state = os.environ.get("SPEC_NGRAM_FIX_GLA", "0") == "1"\n', 1)
    changes += 1; print("added __init__ flag")

# 2) commit GLA state after verify
anchor2 = "            accept_lens = verify_input.accept_length\n"
addition = anchor2 + (
    "            if self._fix_gla_state:\n"
    "                _mr = self.target_worker.model_runner\n"
    "                if getattr(_mr, \"minicpm_hybrid_config\", None) is not None:\n"
    "                    _mr.attn_backend.update_simple_gla_state_after_target_verify(\n"
    "                        accepted_steps=verify_input.accept_length,\n"
    "                        mamba_track_indices=getattr(batch, \"mamba_track_indices\", None),\n"
    "                        mamba_steps_to_track=None,\n"
    "                    )\n"
)
if "update_simple_gla_state_after_target_verify" not in src:
    assert anchor2 in src, "anchor2 (accept_lens) not found"
    src = src.replace(anchor2, addition, 1)
    changes += 1; print("added GLA-state commit after verify")

if changes == 0:
    print("already patched")
else:
    open(F, "w").write(src); print(f"wrote {changes} change-groups")

try:
    py_compile.compile(F, doraise=True); print("py_compile OK")
except Exception as e:
    print("py_compile FAILED, restoring:", e); shutil.copy(bak, F); sys.exit(1)
