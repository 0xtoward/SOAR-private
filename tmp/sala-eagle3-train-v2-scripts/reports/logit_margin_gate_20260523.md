# EAGLE Verify Logit Margin Gate 2026-05-23

## Purpose

This probe checks whether the current MiniCPM-SALA EAGLE force-reject mismatches are only harmless floating-point tail differences, or whether target-verify logits actually change greedy `argmax`.

If output mismatch happens while root-row `argmax` stays identical, we should inspect output plumbing or consider a looser non-exact diagnostic. If output mismatch happens at the same token where verify `argmax` changes, exact-match is not over-strict: target verify is making a different greedy decision.

## Run

- remote host: `autodl-container-42rqpz31dq-e7715add`
- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_logit_margin_5prompt_extend_20260523_040914`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_040914_logit_margin_5prompt_extend`
- SGL repo: `/root/autodl-tmp/codex-drafts/sglang-sala-tree-verify`
- backend: `--attention-backend triton`
- CUDA graph: disabled
- deterministic: enabled
- head: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800`
- spec config: `EAGLE3`, `steps=2`, `topk=4`, `draft_tokens=9`
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_EXTEND_KERNEL=1`
- decode-equivalent all-row fallback: disabled
- force reject: enabled

## Result

| prompt | classification | output mismatch | verify argmax mismatch | first nonzero logit diff | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `known_index59_repeat` | `outputs_exact` |  |  | 6 | Logits drift, but top1 stayed stable. |
| `natural_en` | `verify_argmax_changed_at_output_mismatch` | 90 | 90 | 6 | Real greedy decision changed. |
| `natural_zh` | `verify_argmax_changed_at_output_mismatch` | 23 | 23 | 13 | Real greedy decision changed. |
| `code_structured` | `outputs_exact` |  |  | 2 | Logits drift, but top1 stayed stable. |
| `copy_heavy` | `verify_argmax_changed_at_output_mismatch` | 65 | 65 | 1 | Real greedy decision changed. |

## Mismatch Rows

| prompt | token idx | baseline out | spec out | baseline argmax | verify argmax | baseline margin | verify margin | max abs diff | mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `natural_en` | 90 | 1426 | 1384 | 1426 | 1384 | 0.0625 | 0.0 | 0.1640625 | 0.0220641 |
| `natural_zh` | 23 | 59455 | 35503 | 59455 | 35503 | 0.0625 | 0.0 | 0.1796875 | 0.0171778 |
| `copy_heavy` | 65 | 72 | 1384 | 72 | 1384 | 0.0 | 0.0625 | 0.1875 | 0.0317410 |

## Conclusion

The exact gate is not the primary problem here. For 3 of 5 prompts, the speculative force-reject path changes target-verify greedy `argmax` at the same index where generated output diverges. That means the current tree-extend verify path is not lossless even when no draft token is accepted.

Therefore we should not loosen exact-match as the fix. The next exactness work should continue on verify equivalence, with the decode-equivalent all-row target-verify path as the correctness oracle and the tree-extend path as the failing implementation.

The likely next kernel/debug target remains the attention implementation difference between path-aware `extend_attention_fwd_unified(q_len=1)` and decode grouped attention, not EAGLE head quality or SimpleGLA commit. This is supported by the separate deterministic all-row decode-attention sweep passing topk1/topk2/topk4 force-reject and accept cases.

## Exact Oracle Recheck

After the margin gate, the same 5 prompts were re-run with the env-gated all-row decode-attention fallback:

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_exact_decode_fallback_5prompt_20260523_041922`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_041922_exact_decode_fallback_5prompt`
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true`
- tree-extend path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_EXTEND_KERNEL=false`
- config: `steps=2`, `topk=4`, `draft_tokens=9`

All baseline, force-reject, and accept cases were exact on all 5 prompts:

| case | exact prompts | avg gen per verify | max accept | tok/s range |
| --- | ---: | --- | ---: | --- |
| `force_reject_s2_topk4_draft9` | 5 / 5 | about 1.01 | 0 | 3.28-3.54 |
| `accept_s2_topk4_draft9` | 5 / 5 | 1.09-1.88 | 2 | 3.79-6.61 |

This confirms we have a correctness oracle. The remaining engineering work is to make the verify attention path use decode-equivalent math without paying the full current fallback overhead.

## Default Deterministic Full Matrix After Patch

The decode-equivalent path is now selected by default for MiniCPM-SALA target verify when deterministic inference is enabled. No `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL` env var was set in this run.

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_exact_default_det_fullmatrix_20260523_043636`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_043636_exact_default_det_fullmatrix`
- deterministic: `--enable-deterministic-inference`
- explicit tree env: unset

All cases were exact on all 5 prompts:

| family | configs covered | force-reject exact | accept exact |
| --- | --- | ---: | ---: |
| topk1 chain | `s1_topk1_draft2`, `s3_topk1_draft4`, `s5_topk1_draft6`, `s5_topk1_draft8` | 20 / 20 | 20 / 20 |
| topk2 tree | `s2_topk2_draft5` | 5 / 5 | 5 / 5 |
| topk4 tree | `s2_topk4_draft9` | 5 / 5 | 5 / 5 |

Representative acceptance ranges:

| case | avg_gen_per_verify range | max_accept | tok/s range |
| --- | --- | ---: | --- |
| `accept_s1_topk1_draft2` | 1.05-1.43 | 1 | 9.42-12.70 |
| `accept_s3_topk1_draft4` | 1.07-1.68 | 3 | 7.17-11.56 |
| `accept_s5_topk1_draft6` | 1.07-1.71 | 5 | 5.99-9.65 |
| `accept_s2_topk2_draft5` | 1.08-1.75 | 2 | 5.26-8.56 |
| `accept_s2_topk4_draft9` | 1.09-1.88 | 2 | 3.85-6.62 |

This satisfies the deterministic correctness gate for force-reject, topk1 chain, and topk2/topk4 tree. Performance remains intentionally slow because the current safe path calls decode-equivalent grouped attention for every verify tree row.

## Non-Deterministic Controls

Two follow-up controls were run after the deterministic full matrix to check whether correctness can be recovered by only changing target-verify attention, without enabling deterministic inference.

### Default Non-Deterministic Tree Path

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_nondet_default_topk4_20260523_050113`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_050113_nondet_default_topk4`
- deterministic: disabled
- explicit tree env: unset
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: `natural_en`, `natural_zh`, `copy_heavy`

Result: all 3 prompts mismatched in force-reject and accept mode.

| case | exact prompts | first mismatches | avg gen per verify | max accept | tok/s range |
| --- | ---: | --- | --- | ---: | --- |
| `force_reject_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@23`, `copy_heavy@65` | about 1.01 | 0 | 3.93-4.40 |
| `accept_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@23`, `copy_heavy@65` | 1.09-1.81 | 2 | 4.71-7.90 |

### Non-Deterministic With Decode-Equivalent Tree Attention

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_nondet_decodeeq_topk4_20260523_050515`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_050515_nondet_decodeeq_topk4`
- deterministic: disabled
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true`
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: `natural_en`, `natural_zh`, `copy_heavy`

Result: forcing decode-equivalent tree attention alone was not sufficient. Force-reject still mismatched on all 3 prompts.

| case | exact prompts | first mismatches | avg gen per verify | max accept | tok/s range |
| --- | ---: | --- | --- | ---: | --- |
| `force_reject_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@33`, `copy_heavy@65` | about 1.01 | 0 | 3.75-4.12 |
| `accept_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@33`, `copy_heavy@65` | 1.04-1.81 | 2 | 4.15-7.26 |

This means the current exact fix needs the deterministic/batch-invariant-safe target path, not just the decode-equivalent tree attention switch. The remaining performance work should not remove the exact gate; it should narrow which deterministic/batch-invariant ops are required for MiniCPM-SALA target verify and then replace the slow per-row decode-equivalent attention with a faster equivalent implementation.

### Non-Deterministic With Decode-Equivalent Attention And DeepGEMM Disabled

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_nondet_decodeeq_nodeepgemm_topk4_20260523_051009`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_051009_nondet_decodeeq_nodeepgemm_topk4`
- deterministic: disabled
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true`
- extra env: `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: `natural_en`, `natural_zh`, `copy_heavy`

Result: disabling DeepGEMM batch-invariant ops without `--enable-deterministic-inference` was still not sufficient.

| case | exact prompts | first mismatches | avg gen per verify | max accept | tok/s range |
| --- | ---: | --- | --- | ---: | --- |
| `force_reject_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@33`, `copy_heavy@65` | about 1.01 | 0 | 3.68-4.10 |
| `accept_s2_topk4_draft9` | 0 / 3 | `natural_en@90`, `natural_zh@33`, `copy_heavy@65` | 1.04-1.81 | 2 | 4.23-7.32 |

The failed `nodeepgemm` run showed that just disabling DeepGEMM is not enough; the batch-invariant operator replacements must actually be enabled.

### Non-Deterministic With Decode-Equivalent Attention And Forced Batch-Invariant Ops

To isolate the useful part of `--enable-deterministic-inference`, an env-gated code patch was added in `model_runner.py`:

```text
SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1
```

This calls `enable_batch_invariant_mode()` without enabling the full deterministic serving mode. It globally installs batch-invariant CUDA implementations for `aten::mm`, `aten::addmm`, `aten::bmm`, `aten::_log_softmax`, and `aten::mean.dim`.

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_nondet_decodeeq_forcebio_topk4_20260523_051545`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_051545_nondet_decodeeq_forcebio_topk4`
- deterministic: disabled
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true`
- extra env:
  - `SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1`
  - `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: `natural_en`, `natural_zh`, `copy_heavy`

Result: this recovered exactness for all tested prompts without `--enable-deterministic-inference`.

| case | exact prompts | avg gen per verify | max accept | tok/s range |
| --- | ---: | --- | ---: | --- |
| `force_reject_s2_topk4_draft9` | 3 / 3 | about 1.01 | 0 | 3.27-3.53 |
| `accept_s2_topk4_draft9` | 3 / 3 | 1.04-1.81 | 2 | 3.70-6.38 |

Current exact condition is now narrower than full deterministic mode:

```text
SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1
+ SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
+ MiniCPM-SALA target-verify tree rows using decode-equivalent attention
```

`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` alone does not recover exactness in this matrix. The next exactness/performance step is to expand this narrower mode from the 3-prompt topk4 smoke to the full topk1/topk2/topk4 matrix, then replace the slow per-row decode-equivalent attention with a faster equivalent implementation.

## Forced Batch-Invariant Full Matrix

The narrower exact condition above was expanded from the 3-prompt topk4 smoke to the full prompt/config matrix, still without `--enable-deterministic-inference`.

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_forcebio_fullmatrix_20260523_052119`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_052119_forcebio_fullmatrix/eagle_forcebio_fullmatrix_light_summary.md`
- deterministic: disabled
- verify path: `SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true`
- extra env:
  - `SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1`
  - `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
- prompts: `known_index59_repeat`, `natural_en`, `natural_zh`, `code_structured`, `copy_heavy`

All cases were exact against the Triton baseline output ids:

| case | exact | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_triton` | 5/5 |  |  |  | 76.11 |
| `force_reject_s1_topk1_draft2` | 5/5 | 1.00 | 1 | 0.50 | 8.79 |
| `accept_s1_topk1_draft2` | 5/5 | 1.24 | 2 | 0.62 | 11.01 |
| `force_reject_s3_topk1_draft4` | 5/5 | 1.00 | 1 | 0.25 | 6.92 |
| `accept_s3_topk1_draft4` | 5/5 | 1.34 | 4 | 0.33 | 9.44 |
| `force_reject_s5_topk1_draft6` | 5/5 | 1.00 | 1 | 0.17 | 5.58 |
| `accept_s5_topk1_draft6` | 5/5 | 1.34 | 6 | 0.23 | 7.67 |
| `force_reject_s5_topk1_draft8` | 5/5 | 1.00 | 1 | 0.17 | 5.68 |
| `accept_s5_topk1_draft8` | 5/5 | 1.34 | 6 | 0.23 | 7.68 |
| `force_reject_s2_topk2_draft5` | 5/5 | 1.00 | 1 | 0.20 | 4.79 |
| `accept_s2_topk2_draft5` | 5/5 | 1.38 | 3 | 0.28 | 6.76 |
| `force_reject_s2_topk4_draft9` | 5/5 | 1.00 | 1 | 0.11 | 3.51 |
| `accept_s2_topk4_draft9` | 5/5 | 1.47 | 3 | 0.16 | 5.15 |

This confirms the exactness fix does not require full deterministic serving mode. It does require:

```text
1. Batch-invariant operator replacements enabled.
2. DeepGEMM disabled inside the batch-invariant mm replacement.
3. MiniCPM-SALA target-verify tree attention using the decode-equivalent path.
```

The speed is still not production-worthy: speculative accept cases remain slower than the baseline in this 5-prompt probe because the current safe attention path is intentionally conservative and the EAGLE head acceptance is low. The next work should therefore stay exact-first but move from correctness fallback to a faster decode-equivalent tree verify implementation, then re-run official eval/throughput.

## Batch-Invariant Operator Ablation

The forced batch-invariant mode above was then ablated by registering only subsets of the operator overrides. This identifies which part of deterministic/batch-invariant mode is actually required for the observed non-deterministic mismatch.

- remote root: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_batch_invariant_op_ablation_20260523_055208`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_055208_batch_invariant_op_ablation/ablation_summary.md`
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: `natural_en`, `natural_zh`, `copy_heavy`
- deterministic: disabled
- target verify attention: decode-equivalent
- DeepGEMM inside batch-invariant mm: disabled

Result:

| variant | force-reject exact | accept exact | interpretation |
| --- | ---: | ---: | --- |
| all overrides on | 3/3 | 3/3 | exact reference |
| no `aten::mm` override | 2/3 | 2/3 | mismatch returns at `natural_zh@23` |
| no `aten::addmm` override | 3/3 | 3/3 | not required for this failure |
| no `aten::bmm` override | 3/3 | 3/3 | not required for this failure |
| no `aten::_log_softmax` override | 3/3 | 3/3 | not required for this failure |
| no `aten::mean.dim` override | 3/3 | 3/3 | not required for this failure |

To close the interaction gap, a follow-up `only_mm` run registered only the `aten::mm` batch-invariant override and disabled `addmm`, `bmm`, `log_softmax`, and `mean` overrides:

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_batch_invariant_only_mm_20260523_061518`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_061518_batch_invariant_only_mm/eagle_forcebio_fullmatrix_light_summary.md`

`only_mm` was exact:

| case | exact | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_triton` | 3/3 |  |  |  | 80.10 |
| `force_reject_s2_topk4_draft9` | 3/3 | 1.00 | 1 | 0.11 | 3.55 |
| `accept_s2_topk4_draft9` | 3/3 | 1.34 | 3 | 0.15 | 4.76 |

This narrows the deterministic vs non-deterministic correctness difference to batch-shape-sensitive `aten::mm`. In this probe, `aten::mm` batch-invariant replacement is both necessary (`no_mm` fails) and sufficient (`only_mm` passes) when combined with decode-equivalent target-verify tree attention.

Current narrow exact condition:

```text
SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1
SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
SGLANG_BATCH_INVARIANT_OPS_REGISTER_MM=1
SGLANG_BATCH_INVARIANT_OPS_REGISTER_ADDMM=0
SGLANG_BATCH_INVARIANT_OPS_REGISTER_BMM=0
SGLANG_BATCH_INVARIANT_OPS_REGISTER_LOG_SOFTMAX=0
SGLANG_BATCH_INVARIANT_OPS_REGISTER_MEAN=0
SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true
```

The remaining suspected mechanism is shape-dependent `mm` accumulation in target verify: normal decode and verify tree rows execute different batch/token shapes, which can change bf16 logits enough to flip argmax on low-margin tokens. Exactness is restored by using a batch-invariant `mm` implementation and the decode-equivalent target-verify attention path.

## Only-MM Full Matrix

The narrow condition above was expanded from the 3-prompt topk4 probe to the full 5-prompt topk1/topk2/topk4 matrix.

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_onlymm_fullmatrix_20260523_062149`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_062149_onlymm_fullmatrix/eagle_forcebio_fullmatrix_light_summary.md`
- deterministic: disabled
- target verify attention: decode-equivalent
- batch-invariant operators: only `aten::mm`
- config matrix:
  - `s1_topk1_draft2`: `steps=1`, `topk=1`, `draft_tokens=2`
  - `s3_topk1_draft4`: `steps=3`, `topk=1`, `draft_tokens=4`
  - `s5_topk1_draft6`: `steps=5`, `topk=1`, `draft_tokens=6`
  - `s5_topk1_draft8`: `steps=5`, `topk=1`, `draft_tokens=8`
  - `s2_topk2_draft5`: `steps=2`, `topk=2`, `draft_tokens=5`
  - `s2_topk4_draft9`: `steps=2`, `topk=4`, `draft_tokens=9`

All cases were exact against the Triton baseline output ids:

| case | exact | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_triton` | 5/5 |  |  |  | 79.83 |
| `force_reject_s1_topk1_draft2` | 5/5 | 1.00 | 1 | 0.50 | 9.13 |
| `accept_s1_topk1_draft2` | 5/5 | 1.24 | 2 | 0.62 | 11.46 |
| `force_reject_s3_topk1_draft4` | 5/5 | 1.00 | 1 | 0.25 | 7.19 |
| `accept_s3_topk1_draft4` | 5/5 | 1.34 | 4 | 0.33 | 9.50 |
| `force_reject_s5_topk1_draft6` | 5/5 | 1.00 | 1 | 0.17 | 5.88 |
| `accept_s5_topk1_draft6` | 5/5 | 1.34 | 6 | 0.23 | 7.91 |
| `force_reject_s5_topk1_draft8` | 5/5 | 1.00 | 1 | 0.17 | 5.79 |
| `accept_s5_topk1_draft8` | 5/5 | 1.34 | 6 | 0.23 | 7.84 |
| `force_reject_s2_topk2_draft5` | 5/5 | 1.00 | 1 | 0.20 | 5.05 |
| `accept_s2_topk2_draft5` | 5/5 | 1.38 | 3 | 0.28 | 7.03 |
| `force_reject_s2_topk4_draft9` | 5/5 | 1.00 | 1 | 0.11 | 3.55 |
| `accept_s2_topk4_draft9` | 5/5 | 1.47 | 3 | 0.16 | 5.27 |

This run is the current correctness gate pass for MiniCPM-SALA EAGLE speculative decoding in Triton mode:

```text
temperature=0
baseline normal decode vs EAGLE speculative output_ids exact-match
force-reject exact
topk1 chain exact
topk2/topk4 tree exact
deterministic inference disabled
```

The deterministic vs non-deterministic root cause is now classified as:

```text
Primary: shape-dependent `aten::mm` accumulation in target verify.
Secondary required condition: target-verify tree full attention must use decode-equivalent semantics.
Not implicated by this matrix: accepted state commit, SimpleGLA tree parent-state branching, hot-vocab token map, addmm/bmm/log_softmax/mean batch-invariant overrides.
```

## Default Exact Mode Full Matrix

The narrow exact condition was then made the MiniCPM-SALA target-runner default:

```text
SGLANG_MINICPM_SALA_EXACT_MODE defaults to 1
MiniCPM-SALA target runners enable only the `aten::mm` batch-invariant override
DeepGEMM is disabled inside the batch-invariant mm replacement
MiniCPM-SALA speculative target verify uses decode-equivalent Triton tree attention
```

The full matrix was re-run with all exact-related env vars explicitly unset to confirm the code default, not the shell environment, is carrying correctness.

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_default_exact_fullmatrix_20260523_072140`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_072140_default_exact_fullmatrix/eagle_forcebio_fullmatrix_light_summary.md`
- deterministic: disabled
- explicit exact env: unset
- config matrix:
  - `s1_topk1_draft2`: `steps=1`, `topk=1`, `draft_tokens=2`
  - `s3_topk1_draft4`: `steps=3`, `topk=1`, `draft_tokens=4`
  - `s5_topk1_draft6`: `steps=5`, `topk=1`, `draft_tokens=6`
  - `s5_topk1_draft8`: `steps=5`, `topk=1`, `draft_tokens=8`
  - `s2_topk2_draft5`: `steps=2`, `topk=2`, `draft_tokens=5`
  - `s2_topk4_draft9`: `steps=2`, `topk=4`, `draft_tokens=9`

All cases were exact against the Triton baseline output ids:

| case | exact | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_triton` | 5/5 |  |  |  | 81.18 |
| `force_reject_s1_topk1_draft2` | 5/5 | 1.00 | 1 | 0.50 | 9.12 |
| `accept_s1_topk1_draft2` | 5/5 | 1.24 | 2 | 0.62 | 11.30 |
| `force_reject_s3_topk1_draft4` | 5/5 | 1.00 | 1 | 0.25 | 7.20 |
| `accept_s3_topk1_draft4` | 5/5 | 1.34 | 4 | 0.33 | 9.61 |
| `force_reject_s5_topk1_draft6` | 5/5 | 1.00 | 1 | 0.17 | 5.85 |
| `accept_s5_topk1_draft6` | 5/5 | 1.34 | 6 | 0.23 | 8.05 |
| `force_reject_s5_topk1_draft8` | 5/5 | 1.00 | 1 | 0.17 | 5.86 |
| `accept_s5_topk1_draft8` | 5/5 | 1.34 | 6 | 0.23 | 8.03 |
| `force_reject_s2_topk2_draft5` | 5/5 | 1.00 | 1 | 0.20 | 5.01 |
| `accept_s2_topk2_draft5` | 5/5 | 1.38 | 3 | 0.28 | 6.91 |
| `force_reject_s2_topk4_draft9` | 5/5 | 1.00 | 1 | 0.11 | 3.57 |
| `accept_s2_topk4_draft9` | 5/5 | 1.47 | 3 | 0.16 | 5.31 |

## Scoped Default Smoke

One final patch narrowed the target-verify decode-equivalent attention default so it is not a global Triton speculative behavior. The default decode-equivalent path is now selected by MiniCPM-SALA exact mode or by explicit env; non-MiniCPM-SALA models are not forced onto this conservative path unless the env is set.

After syncing that patch to `rtx6000-1`, a topk4 smoke was re-run with all exact env vars unset:

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_default_exact_scoped_smoke_20260523_074150`
- local summary: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_074150_default_exact_scoped_smoke/eagle_forcebio_fullmatrix_light_summary.md`
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- deterministic: disabled
- explicit exact env: unset

Result:

| case | exact | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_triton` | 5/5 |  |  |  | 104.69 |
| `force_reject_s2_topk4_draft9` | 5/5 | 1.00 | 1 | 0.11 | 3.58 |
| `accept_s2_topk4_draft9` | 5/5 | 1.50 | 3 | 0.16 | 5.31 |

This confirms the scoped default patch preserves MiniCPM-SALA exactness while avoiding a broad default behavior change for other Triton speculative users.

## Current Decision

The logit-margin probe ruled out loosening exact-match as the fix: failing prompts changed target-verify `argmax`, not just low-order logit bits. The current correctness fix is therefore code-side:

```text
MiniCPM-SALA target runner exact mode
+ only `aten::mm` batch-invariant replacement
+ DeepGEMM disabled for that replacement
+ decode-equivalent Triton target-verify tree attention
+ SimpleGLA target-verify cu_seqlens and parent-state path retained
```

This is a correctness gate pass, not a production speed result. The remaining work should optimize this exact path rather than relaxing the gate.
