# FlashInfer Backend Probe 2026-05-23

## Question

Could `--attention-backend flashinfer` avoid the MiniCPM-SALA EAGLE exactness issue without the Triton decode-equivalent verify fallback?

The test compares FlashInfer no-spec baseline output ids against FlashInfer EAGLE speculative output ids. This avoids mixing Triton baseline with FlashInfer speculative output.

## Harness Change

`trace_eagle_deterministic_tree_sweep.sh` now accepts:

```text
ATTENTION_BACKEND=triton|flashinfer
```

and records the actual backend in `case_manifest.jsonl`.

## Run A: FlashInfer Native, SALA Exact Mode Disabled

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_flashinfer_native_noexact_topk4_20260523_083249`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_083249_flashinfer_native_noexact`
- backend: `flashinfer`
- exact mode: disabled with `SGLANG_MINICPM_SALA_EXACT_MODE=0`
- deterministic: disabled
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: 5 fixed smoke prompts

Result:

| case | exact | mismatch count | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_flashinfer` | 5/5 | 0 |  |  |  | 141.20 |
| `force_reject_s2_topk4_draft9` | 3/5 | 2 | 1.00 | 1 | 0.11 | 4.25 |
| `accept_s2_topk4_draft9` | 3/5 | 2 | 1.41 | 3 | 0.16 | 6.06 |

First force-reject mismatches:

| prompt | index | baseline token | spec token |
| --- | ---: | ---: | ---: |
| `known_index59_repeat` | 48 | 1377 | 6603 |
| `copy_heavy` | 4 | 22997 | 13115 |

Interpretation: FlashInfer native speculative verify is not lossless here. Because force-reject mismatches, this is target verify / backend / position / cache metadata territory, not draft-head acceptance.

## Run B: FlashInfer With Default MiniCPM-SALA Exact Mode

- remote run dir: `/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/eval/eagle_flashinfer_default_exact_topk4_20260523_083704`
- local copy: `/Users/ql/cursor/openbmb/tmp/eagle_spec_snapshots/20260523_083704_flashinfer_default_exact`
- backend: `flashinfer`
- exact mode: default MiniCPM-SALA exact mode
- deterministic: disabled
- config: `steps=2`, `topk=4`, `draft_tokens=9`
- prompts: 5 fixed smoke prompts

Result:

| case | exact | mismatch count | avg accept len | max accept len | avg accept rate | avg TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_flashinfer` | 5/5 | 0 |  |  |  | 100.44 |
| `force_reject_s2_topk4_draft9` | 3/5 | 2 | 1.00 | 1 | 0.11 | 3.76 |
| `accept_s2_topk4_draft9` | 3/5 | 2 | 1.53 | 3 | 0.17 | 5.86 |

First force-reject mismatches:

| prompt | index | baseline token | spec token |
| --- | ---: | ---: | ---: |
| `known_index59_repeat` | 59 | 74 | 2131 |
| `natural_zh` | 23 | 59455 | 35503 |

Interpretation: MiniCPM-SALA exact mm mode alone does not make FlashInfer EAGLE verify exact. This suggests the remaining mismatch is in the FlashInfer target-verify backend path itself, not only in shape-dependent `aten::mm`.

## Decision

FlashInfer does not directly bypass the current MiniCPM-SALA EAGLE exactness problem.

The current correctness-safe backend remains Triton with the MiniCPM-SALA exact path:

```text
MiniCPM-SALA target runner exact mode
+ only `aten::mm` batch-invariant replacement
+ DeepGEMM disabled for that replacement
+ decode-equivalent Triton target-verify tree attention
```

FlashInfer may still be useful later for speed work, but it cannot replace the Triton exact oracle today. Any FlashInfer route needs its own target-verify equivalence fix before it can be used for lossless speculative decoding.
