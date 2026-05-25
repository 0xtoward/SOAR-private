# NGRAM spec GLA-state-after-verify fix (MiniCPM-SALA, 2026-05-26)

## The bug
`spec_engine/.../speculative/ngram_worker.py` **never advanced the GLA recurrent state
after target-verify.** The EAGLE worker does this (`eagle_worker.py:910` →
`attn_backend.update_simple_gla_state_after_target_verify(...)`), but the NGRAM worker
omitted it entirely → the GLA temporal state was never rolled to the accepted position →
generation diverged into garbage.

Symptom (eval30 = 30 balanced RULER samples, conc=1, cap32K, NVFP4 4o6):
- no-spec (submission engine): **acc 83.0%**
- NGRAM eager (broken): **17.7%** ; NGRAM cudagraph (broken): **22.6%**
  (high accept_len but garbage output = verify accepts, but committed state is stale)

## The fix
Port EAGLE's post-verify state commit into the ngram worker's `is_target_verify` branch,
with `accepted_steps = verify_input.accept_length`.

Why `accept_length` (not `accept_length-1`): ngram's
`accept_length = (accepted_indices != -1).sum(dim=1) - 1` ≡ eagle's `accept_length_per_req_cpu`
(eagle computes `accepted_steps = (accept_length_per_req_cpu + 1) - 1 = accept_length_per_req_cpu`).
intermediate_ssm[step] is written during the verify forward (hybrid_linear_attn_backend.py:1909);
committing intermediate_ssm[accept_length] = state after the last accepted draft.

Opt-in: env `SPEC_NGRAM_FIX_GLA=1` (default off). Applied via `patch_ngram_fix_gla.py`
(idempotent, backup `.bak_fixgla`, py_compile-verified).

### Result
acc **22→77 (cudagraph), 17→76 (eager)** — recovered +55. NGRAM spec now works.

## Residual ~6pt (77 vs 83) — irreducible on sm120
Scattered per-sample divergence (spec sometimes BETTER and sometimes WORSE than no-spec →
genuine output difference on ~5-6/30, not task/length-localized). Probed
`SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1` → acc stayed ~73-77 (did NOT close) → residual is
GLA chunked-verify-vs-recurrent-decode FP drift, not matmul batch-variance. Not worth a
kernel rewrite.

## SOAR speed/gate verdict (eval30, end-to-end TPS proxy, cap512)
| | S1(c1) | S8(c8) | Smax(c32) | weighted 0.4/0.3/0.3 | acc |
|---|---|---|---|---|---|
| naive (no-spec)   | 58.5 | 110.3 | 125.7 | **94.2** | 83 |
| always-spec       | 62.3 | 103.6 | 102.8 | 86.8 (net-NEG) | 77 |
| gated (spec@S1,   | 62.3 | 110.3 | 125.7 | **95.7 (+1.6%)** | 83* |
|  no-spec@c≥8)     |      |       |       |          |     |

Spec wins only S1 (weight-bound); loses S8/Smax (compute-bound, verify overhead). always-spec
is net-negative. A batch-size gate (spec only at #running-req=1) WOULD win +1.6% AND preserve
acc — because the official acc eval runs at `--concurrency 8` (run_medium_eval.sh:63 /
run_quant_fast_medium.sh:210), where the gate uses no-spec → acc 83.

**BUT** realizing the runtime gate needs plain-decode-at-high-batch inside the spec engine =
the known dead-end: `ScheduleBatch.prepare_for_decode` early-returns for spec engines (layer 4,
spoofable), and the scheduler's spec result-commit expects committed tokens in the verify
buffer, not a plain `next_token_ids` (layer 5 → infinite-gen, previously abandoned).

**Recommendation:** +1.6% on a SECONDARY lever is not worth re-entering the multi-layer
plain-decode dead-end nor risking the proven NVFP4 98.44 mainline. Keep no-spec. This GLA-state
fix is the durable, reusable asset — it makes NGRAM spec correct, valuable for any future
short-ctx target where spec wins broadly.
