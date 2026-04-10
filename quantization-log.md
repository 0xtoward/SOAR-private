# Quantization Log

Note: historical `fast`, `fast`, and `v2` labels below are obsolete. The only active quick gate is `fast`, backed by `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`.

## Environment Note

- Local planning / packaging / notes:
  - `/Users/ql/cursor/openbmb`
- Remote quant execution:
  - `rtx6000-2:/root/autodl-tmp`
- Codex skill definitions:
  - `/Users/ql/.codex/skills`

## 2026-04-09 Current State

- Primary active route:
  - `ModelOpt NVFP4 MLP-only`
  - local curated calibration set at `16K`
  - current submission draft:
    - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft`
- Latest NVFP4 submission blocker that was fixed:
  - `device_map="auto"` required `accelerate` on the platform
  - fixed by loading normally and then moving the model to CUDA explicitly
- Fresh NVFP4 artifact:
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft.tar.gz`
  - sha256: `2ae2f4809c28d6a18b0750ea89b5de957413d3054bc99cfdd2fc45a6ca328001`

## Secondary Route

- `W4A16 + GPTQ + Marlin`
- detailed route note:
  - `/Users/ql/cursor/openbmb/marlin-route-log.md`
- current judgment:
  - technically viable as a runtime fallback
  - not yet a primary scoring candidate
  - main blocker is checkpoint quality and submission-safe GPTQ checkpoint preparation

## Current Priorities

1. Keep the NVFP4 route submission-safe and measurable.
2. Improve calibration quality rather than blindly increasing experiment count.
3. Treat Marlin as a secondary hedge until a trustworthy GPTQ checkpoint path exists.

## Rules

- Do not run two heavy GPU jobs at the same time.
- Always log quantization outcomes, blockers, and new artifact hashes here or in the route-specific note.
- Prefer cheap smoke or mini validation before expensive full runs.

## 2026-04-09 Next Runnable Candidate

- Why this candidate now:
  - the current most credible quantization route is still `NVFP4 MLP-only`, not Marlin
  - a fully exported remote checkpoint already exists:
    - `/root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128`
  - this candidate uses the improved `16K` curated calibration set and does not require another heavy quantization pass before first measurement
  - the fastest way to reduce uncertainty is to bench the best existing quantized checkpoint against the now-repaired harness
- Chosen next step:
  - run the existing `NVFP4 MLP-only curated16k-128` checkpoint through the bench runner
  - use the same serving args as the current stable base path, plus `--quantization modelopt`
  - treat this as the first serious quantization-side measurement on the repaired harness
- Exact next command:
  - `python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py --label nvfp4-curated16k128-smoke --model-path /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128 --quantization modelopt --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --disable-radix-cache --disable-cuda-graph --dense-as-sparse --profile smoke`
- Expected outcome:
  - if this looks preliminarily good, it becomes the quantization route that deserves the first isolated full eval
  - if it regresses badly, the next quant step should be a new `16K` NVFP4 re-quant with more calibration samples, not a switch to Marlin

## 2026-04-09 NVFP4 curated16k-128 stable smoke

- Exact command launched:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label nvfp4-curated16k128-smoke-stable \
  --model-path /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128 \
  --quantization modelopt \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile smoke
```

- Run directory:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_031108_nvfp4-curated16k128-smoke-stable`
- Smoke eval result:
  - `avg_score=32.00%`
  - `pass=1`
  - `part=1`
  - `fail=3`
- Per-task signals:
  - `cwe=0.60`
  - `fwe=0.00`
  - `mcq=0.00`
  - `niah=0.00`
  - `qa=1.00`
- Immediate read:
  - this is only slightly above the earlier base smoke result (`28%`)
  - the route is alive and submission-safe on the stable runtime path
  - but it is not yet a clearly strong quality result
- Current state:
  - `bench_smoke` is still running after the eval phase
  - champion-route guidance from `week3/week4` should be used to decide whether NVFP4 remains an active optimization target or is demoted to fallback after this run

## 2026-04-09 Immediate Post-Smoke Decision Ladder

- Candidate under test:
  - `nvfp4-curated16k128-smoke`
  - remote checkpoint:
    - `/root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128`

- `Pass` outcome:
  - Signal:
    - smoke is clearly competitive with the current base smoke and does not show broad task collapse
    - no obvious load/runtime failure from `quantization=modelopt`
  - Immediate next quantization-side action:
    - freeze this checkpoint as the primary quant route
    - run one isolated full public eval for this exact route before changing calibration or config
  - Exact next command after a pass:
    - `API_BASE=http://127.0.0.1:30001 python /root/autodl-tmp/SOAR-Toolkit/eval_model.py --help`
    - then execute the official full eval flow for this exact checkpoint with no competing tests

- `Mediocre` outcome:
  - Signal:
    - checkpoint loads and runs, but smoke is mixed rather than clearly strong
    - examples: small uplift over the base smoke, unstable stop behavior, or one-task regression without total collapse
  - Immediate next quantization-side action:
    - keep `mlp_only`
    - rerun NVFP4 with the same curated `16K` source but raise local calibration coverage from `128` to `192` or `256`
    - do not switch to Marlin yet
  - Exact next command after a mediocre result:
    - `python3 /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-256 --cfg mlp_only --dtype bfloat16 --local-calib-files /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 256 --max-length 16384 --min-good-batches 96 --overwrite`

- `Fail` outcome:
  - Signal:
    - quantized route is clearly worse than base smoke, fails to load, or shows broad collapse
    - repeated `finish_reason=length` or severe task-wide regression counts as fail
  - Immediate next quantization-side action:
    - do one bounded fallback re-quant before abandoning NVFP4:
    - keep the curated data, but shorten calibration length to `8192` and keep `mlp_only` to reduce calibration-path instability
    - only if that fallback also fails should Marlin be promoted
  - Exact next command after a fail:
    - `python3 /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-only-curated8k-128 --cfg mlp_only --dtype bfloat16 --local-calib-files /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 128 --max-length 8192 --min-good-batches 64 --overwrite`

## 2026-04-09 Draft Packaging Cleanup

- Removed an unnecessary local bytecode artifact from the submission draft:
  - `__pycache__/quantize_modelopt_nvfp4_conservative.cpython-39.pyc`
- Why:
  - it is host-specific noise, not needed for submission
  - carrying local `pyc` output is avoidable packaging risk
- Fixed draft checksum generation:
  - `SHA256SUMS.txt` is now generated with null-delimited paths so files with spaces, commas, or brackets are hashed correctly
- Fresh artifact after cleanup:
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft.tar.gz`
  - sha256: `2fe26ebebccd33a06d471ef77551e26139b98f78a475b0abe186e7cf22ea58fe`

## 2026-04-09 Runtime Blocker Note For Quant Evaluation

- Latest runtime finding:
  - `fuse_topk=True` currently trips a runtime bug in `minicpm_backend.py` because `max_running_requests` is `None` and the fused path assumes an integer
- Quantization-side impact:
  - this does not change the NVFP4 decision ladder
  - it does change the comparison protocol for the first quant smoke
- Rule for the first quant smoke comparison:
  - do not enable `--fuse-topk` on the first `nvfp4-curated16k128-smoke`
  - first establish a clean quant-vs-base comparison on the current stable runtime path
  - only after the runtime bug is fixed should `fuse_topk` be reintroduced for a later quantized A/B
- Quant action that remains queued:
  - once runtime is healthy, run the existing `nvfp4-curated16k128-smoke` command exactly as logged above, with no `--fuse-topk`

## 2026-04-09 NVFP4 Stable Smoke Classification

- Measured result:
  - `avg_score=32.00%`
  - `pass=1 part=1 fail=3`
  - per-task signals:
    - `cwe=0.60`
    - `fwe=0.00`
    - `mcq=0.00`
    - `niah=0.00`
    - `qa=1.00`
- Baseline comparison:
  - earlier base smoke was `28.00%`
  - this is only a small uplift and does not show broad task robustness
- Decision-ladder classification:
  - `Mediocre`, but weak mediocre rather than promising mediocre
- Champion-guided judgment:
  - do not spend the next free GPU window on another NVFP4 calibration iteration
  - demote current `NVFP4 MLP-only curated16k-128` from primary push to fallback candidate
  - move the next primary quantization push to the champion-aligned `W4A16 + GPTQ + Marlin` route
- Why:
  - Week 4 explicitly prioritizes `W4A16 + GPTQ + Marlin` as the main route for decode-bandwidth gains
  - this NVFP4 smoke does not show enough accuracy headroom to justify another immediate bounded re-quant ahead of the champion route
  - NVFP4 remains worth keeping as a hedge if the champion route fails to materialize cleanly
- Exact next quantization move after the current bench finishes:
  - prepare and measure the champion route first, using the existing local GPTQ checkpoint and `gptq_marlin` runtime path, with no `fuse_topk`
  - exact first measurement commands are recorded in:
    - `/Users/ql/cursor/openbmb/marlin-route-log.md`
- Deferred fallback only:
  - if the champion route cannot be materialized cleanly or is clearly worse on first smoke, then come back to one bounded NVFP4 re-quant:
    - `python3 /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-256 --cfg mlp_only --dtype bfloat16 --local-calib-files /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 256 --max-length 16384 --min-good-batches 96 --overwrite`

## 2026-04-09 Stock-Base Control Implication For Quant Decisions

- Control path:
  - original weights
  - near-stock SGLang tree built from `git archive HEAD` with `minicpm.py.orig` restored
- Control smoke result from:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_033456_stock-base-fp16-smoke-control/fast_eval.log`

```text
avg_score=28.00%
pass=1
part=1
fail=3
```

- Quant-side read:
  - this exactly matches the earlier working-tree `base-fp16` smoke headline score
  - therefore the current tiny `smoke` harness does not support blaming the fused `minicpm.py` edits for weak base quality
  - quant routing should not pivot based on another model-file rollback theory
- Quant gate after this control:
  - if control validity continues to hold, the next quant experiment remains the queued `local GPTQ first` `W4A16 + GPTQ + Marlin` smoke
  - if a broader stock-base quality check later contradicts the tiny smoke signal, revisit the harness first before spending another heavy quantization cycle

## 2026-04-09 Stock-Base Control Interpretation Rule

- Champion-guided framing:
  - Week 4 says local high-confidence evaluation infrastructure is a prerequisite before route comparisons
  - Week 3 says `fwe/cwe` are semantically weak and should not dominate calibration or accuracy judgment
- If the upcoming stock-base control group also scores low:
  - treat it as a `harness/runtime issue` first if you see empty outputs, repeated `finish_reason=length`, decode crashes, timeout/pathology, or other obvious generation-control failures
  - treat it as a `smoke-noise / representativeness issue` if smoke is low but proxy is materially healthier, or if the weakness is concentrated in `fwe/cwe` while more semantic tasks are still comparatively sane
  - treat it as `MiniCPM-SALA native local baseline is just not high here` only if both smoke and proxy stay low on stock-base with stable outputs and no runtime/pathology signals
- Quantization decision implication:
  - if stock-base control is technically invalid, do not spend the next GPU window on any new quant route
  - if stock-base control is technically valid, compare future quant routes against that baseline rather than assuming the harness is wrong

## 2026-04-09 Stock-Base Control Smoke Result

- Control setup:
  - remote editable `sglang` was switched to `/root/autodl-tmp/sglang-stock/python`
  - that stock tree was built from `git archive HEAD` and then had `minicpm.py.orig` restored into place
- Fast smoke result from:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_033456_stock-base-fp16-smoke-control/fast_eval.log`

```text
avg_score=28.00%
pass=1
part=1
fail=3
```

- Per-task signals:
  - `cwe=0.40`
  - `fwe=1.00`
  - `mcq=0.00`
  - `niah=0.00`
  - `qa=0.00`
- Quantization-side interpretation:
  - this matches the earlier `base-fp16` smoke headline score exactly
  - do not blame the current fused `minicpm.py` working tree for the weak smoke score
  - the control is technically valid enough to keep using smoke as a crude baseline, but not strong enough to crown routes on its own
- Quantization decision update:
  - keep `W4A16 + GPTQ + Marlin` as the next primary quant candidate after the current stock speed smoke resolves
  - do not spend the next free GPU window on another NVFP4 re-quant first
  - if we need a stronger quality read before committing to the next quant route, prefer a stock-base `mini_test.sh` control over another smoke rerun

## 2026-04-09 Quant Gate After Official-Eval Discovery

- Official toolkit eval confirmed:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_model.py`
- Quant-side policy update:
  - because stock-base `fast_test.sh` also lands at `28.00%`, do not use `fast_test.sh` alone to judge whether a quant route is good or bad
  - keep smoke for startup/sanity only
  - hold the queued `gptq_marlin` quality judgment behind one official stock-base `eval_model.py` run
- Quant next-step implication:
  - route priority remains `W4A16 + GPTQ + Marlin`
  - but the first serious comparison after the current speed smoke should reference the official stock-base eval, not the 5-sample smoke score

## 2026-04-09 Quant Hold While Official Stock Eval Runs

- The official stock-baseline eval is now running in remote tmux:
  - `codex-soar:stock-official-eval`
  - run dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Quantization policy during this run:
  - do not launch a new quantized candidate while the official stock baseline is unresolved
  - keep `W4A16 + GPTQ + Marlin` as the queued next quant route
  - judge that route against the official stock result once it lands, not against the 5-sample smoke score alone

## 2026-04-09 Official Stock Eval Running

- Official stock-base quality control is now active in remote `tmux`:
  - session/window: `codex-soar:stock-official-eval`
- Run directory:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Quantization implication while this run is active:
  - do not promote or demote `W4A16 + GPTQ + Marlin` based on the `28.00%` smoke alone
  - wait for the official stock-base eval outcome, then compare the first `gptq_marlin` smoke against that fuller control rather than against the 5-sample smoke gate

## 2026-04-09 Stock-Base Official Eval Running

- Current confirmed state:
  - `stock-base` control smoke remains `28.00%`, consistent with the earlier base FP16 smoke
  - official original-weight eval is currently running in remote tmux window:
    - `stock-official-eval`
  - log directory:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Quantization-side rule while this run is active:
  - do not launch any new heavy quantization or quantized-serve job
  - treat this official stock-base eval as the decisive control for route prioritization
- If the official stock-base eval also ends low:
  - stop treating low smoke as a reason to keep debugging the smoke harness before every quant move
  - accept that the local/official baseline for raw MiniCPM-SALA is simply not high enough to justify more NVFP4 calibration spend right now
  - keep `NVFP4 MLP-only` in fallback/hedge position only
  - shift the primary quant goal to:
    - preserve enough correctness to avoid collapse
    - then win on the Week 4 champion axis of `W4A16 + GPTQ + Marlin` decode efficiency
  - practical route adjustment:
    - do not queue another immediate NVFP4 re-quant after the official stock-base result
    - instead, use the official stock-base result as the comparison anchor for the first `gptq_marlin` smoke/proxy and decide promotion from there

## 2026-04-09 Official Eval Priority For Quant Routes

- New main goal:
  - prioritize `quant route review + official full accuracy eval`
  - do not optimize for speed-first queueing in this phase

- Official full-eval priority:
  1. `W4A16 + GPTQ + Marlin` using the local GPTQ checkpoint `/root/autodl-tmp/models-gptq-w4a16-v19`
  2. `NVFP4 MLP-only curated16k-128` as the current Blackwell-native fallback
  3. one bounded `NVFP4 MLP-only curated16k-256` re-quant only if route 1 fails to materialize cleanly and route 2 is still the strongest surviving hedge

- Why these are the only high-value full-eval candidates right now:
  - route 1 is the Week 4 champion-aligned mainline and has the highest strategic upside
  - route 2 is already materialized, already measured, and is the only current non-Marlin quant route with any positive smoke signal over base
  - route 3 is the only additional NVFP4 spend that is still bounded and hypothesis-driven; anything broader than this is too speculative for a full official eval slot

- Explicitly not worth burning official full eval on right now:
  - public fallback GPTQ repo `ericzhang0328/MiniCPM-SALA-GPTQ-w4a16-ctx4096`
    - prior local smoke was too weak
  - `plain gptq` without Marlin
    - worse strategic fit than the champion route
  - another immediate NVFP4 variant beyond `curated16k-256`
    - too many moving parts before we have a clean official anchor
  - any route that depends on `fuse_topk`
    - current runtime path is known-bad there

## 2026-04-09 Official Full-Eval Result Read Template

- Current official queue:
  - `official_gptq_marlin -> official_nvfp4_curated16k128`

- Read these signals first for each route:
  - did the official eval complete cleanly
  - official correctness / total score relative to the running `stock-base` official eval
  - whether logs show broad task collapse, empty outputs, or length/pathology behavior rather than simple answer errors

- Decision rule:
  - keep pushing `gptq_marlin` if it completes cleanly and its official result is at least competitive with stock-base while clearly ahead of `nvfp4_curated16k128`
  - switch back to `NVFP4` fallback if `gptq_marlin` fails technically, or if its official result is materially worse than both stock-base and `nvfp4_curated16k128`
  - require a third candidate only if both `gptq_marlin` and `nvfp4_curated16k128` are technically valid but neither gives an acceptable official result

- Third-candidate rule:
  - if a third candidate is needed, use only the bounded `NVFP4 MLP-only curated16k-256` re-quant
  - do not open a wider quant search before these two official anchors are understood

## 2026-04-09 Official Full-Eval Result Extraction Template

- After an official full eval finishes, run this on the run directory:
  - `RUN_DIR=/root/autodl-tmp/SOAR-Toolkit/test_results/<run_dir>; rg -n "Average Score|Total Duration|Request failed|ERROR" "$RUN_DIR" -S`
- What to read from it:
  - `Average Score`
  - `Total Duration`
  - whether any `Request failed` or `ERROR` lines appear

## 2026-04-09 Three-Candidate Official Comparison Skeleton

| Signal | stock-base | gptq_marlin | nvfp4 curated16k128 |
|---|---:|---:|---:|
| Average Score |  |  |  |
| Total Duration |  |  |  |
| Request failed / ERROR |  |  |  |
| Technical validity |  |  |  |
| Quant decision | baseline |  |  |

- Fill rule:
  - `Technical validity`: `valid` / `failed`
  - `Quant decision`: `push` / `fallback` / `drop`

## 2026-04-09 Next Bounded NVFP4 Accuracy Candidate

- Script/manifest facts that matter:
  - current active manifest is:
    - `cfg=mlp_only`
    - `dtype=bfloat16`
    - `local_calib_samples=128`
    - `max_length=16384`
    - `min_good_batches=64`
    - `good_batches=128`
    - `skipped_batches=0`
  - `calibration_curated_16k_v1.jsonl` currently contains exactly `128` records and `128` unique texts
- Implication:
  - raising `local_calib_samples` above `128` on the current `calibration_curated_16k_v1.jsonl` is not a real change
  - keeping `bf16` is preferred because the current successful manifest already shows stable `128/128` good batches with zero skips
- Most likely next accuracy-improving NVFP4 candidate:
  - `NVFP4_MLP_WEIGHT_ONLY_CFG`
  - keep the same curated file
  - keep `local_calib_samples=128`
  - keep `max_length=16384`
  - keep `dtype=bfloat16`
- Why this is more worth trying than another `curated16k128 mlp_only` variant first:
  - it is a real change, whereas `128 -> 256` on the current file is not
  - it narrows quantization scope more conservatively than `mlp_only`, which is the most plausible low-risk accuracy gain when accuracy is prioritized over speed
  - it preserves the same long-context calibration distribution and the same stable bf16 quantization path, so the experiment isolates one lever instead of changing several
- Exact command template:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated16k-128 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 128 --max-length 16384 --min-good-batches 64 --overwrite'`

## 2026-04-09 Updated NVFP4 Candidate Ranking After Public-Bucket Review

- New facts:
  - public-set prompt distribution is heavily long-context:
    - `0-4K: 30`
    - `16K-32K: 40`
    - `32K-128K: 80`
    - `max_prompt_tokens=127738`
  - current `calibration_curated_16k_v1.jsonl` has exactly `128` unique texts
  - `TOK_P50=16384` and `TOK_P90=16384`, so the current v1 set is strongly capped by the `16K` slice limit
  - `32K + pg19` v2 calibration generation is already in progress at:
    - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_pg19_v2.jsonl`

- Clear ranking under single-card / eval-machine time limits:
  1. `B) mlp_weight_only + bf16 + 32K + 64 samples`
  2. `A) mlp_weight_only + bf16 + 16K + 128 samples`
  3. `C) mlp_only + bf16 + 32K + 64 samples`

- Why `B` is now the top candidate:
  - it is the smallest change that actually addresses the newly confirmed mismatch between calibration length and public-set length distribution
  - `64 x 32K` keeps total calibration token budget roughly equal to `128 x 16K`, so it is still bounded for a single-card run
  - `mlp_weight_only` remains the safer accuracy-first scope than `mlp_only`
  - this makes `B` both more distribution-correct than `A` and more conservative than `C`

- Why `A` drops to second:
  - it is still cheap and stable
  - but it does not fix the main new problem: the v1 calibration set is length-capped at `16K` even though the public set is dominated by `32K-128K`

- Why `C` is only third:
  - it improves length alignment like `B`
  - but it keeps the broader `mlp_only` quantization scope, which is less accuracy-friendly than `mlp_weight_only`

- Exact command templates:
  - `#1 B`
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated32k64-v2 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_pg19_v2.jsonl --local-calib-samples 64 --max-length 32768 --min-good-batches 48 --overwrite'`
  - `#2 A`
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated16k-128 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 128 --max-length 16384 --min-good-batches 64 --overwrite'`
  - `#3 C`
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-only-curated32k64-v2 --cfg mlp_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_pg19_v2.jsonl --local-calib-samples 64 --max-length 32768 --min-good-batches 48 --overwrite'`

## 2026-04-09 Networkless 32K Branch

- New constraint:
  - external dataset download is currently unreliable, so the next bounded NVFP4 candidate should not depend on `pg19` / other online datasets

- Best networkless NVFP4 candidate:
  - still choose `B` in spirit:
    - `mlp_weight_only + bfloat16 + 32K + fewer samples`
  - but build the `32K` calibration file from `local public set only`

- Why this remains the best networkless first try:
  - it still fixes the main newly confirmed issue: the current v1 calibration set is capped at `16K` while the public set is dominated by `32K-128K`
  - it avoids external downloads entirely
  - it keeps the safer `mlp_weight_only` scope and the already-stable `bf16` quant path
  - `64 x 32K` is still the right bounded token budget target

- Important build detail:
  - for `32K`, do not rely on the script defaults for head/middle/tail because the current defaults would implicitly over-allocate extra budget to the head
  - make the slice explicitly tail-heavy

- Exact networkless command templates:
  - build the new local-public-only 32K calibration file:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /Users/ql/cursor/openbmb/scripts/build_calibration_curated.py --model-path /root/autodl-tmp/models --public-path /root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl --output-path /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_publiconly_v1.jsonl --public-quotas mcq:14,qa:20,niah:20,fwe:5,cwe:5 --open-quotas cnn_dailymail:0,wikitext:0,pg19:0 --max-sample-tokens 32768 --head-tokens 4096 --middle-tokens 4096 --tail-tokens 24576'`
  - then run the bounded NVFP4 quantization candidate:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated32k64-publiconly-v1 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_publiconly_v1.jsonl --local-calib-samples 64 --max-length 32768 --min-good-batches 48 --overwrite'`

## 2026-04-09 Trigger Rule For `calibration_curated_32k_public_v1.jsonl`

- Trigger:
  - only apply this rule after I confirm that `calibration_curated_32k_public_v1.jsonl` has been generated successfully and is readable

- Required lightweight checks before choosing sample count:
  - total record count
  - unique text count
  - token distribution at least for:
    - `TOK_P50`
    - `TOK_P90`
    - `TOK_MAX`
  - how many samples actually land in:
    - `16K-32K`
    - `<16K`

- Decision rule for `local_calib_samples`:
  - keep `64` if the file has at least `64` unique texts and `TOK_P50` is already near `32K`
  - raise to `80` if the file has clear extra unique long samples beyond `64` and `TOK_P50/TOK_P90` both still stay meaningfully above the old `16K` cap
  - raise to `96` only if:
    - the file still has `96+` unique texts
    - the added samples are mostly still in the `16K-32K` bucket
    - and the token budget increase looks worthwhile relative to single-card runtime

- Anti-pattern rule:
  - do not raise from `64` to `80/96` just because the file is longer on disk
  - only raise if the extra samples are genuinely additional `16K-32K` examples rather than duplicates or mostly shorter prompts

- Expected follow-up:
  - once the file is confirmed, first do the lightweight distribution check above
  - then choose exactly one of:
    - `mlp_weight_only + bf16 + 32K + 64`
    - `mlp_weight_only + bf16 + 32K + 80`
    - `mlp_weight_only + bf16 + 32K + 96`
  - do not queue multiple 32K sample-count variants at once

## 2026-04-09 Preliminary Read For `calibration_curated_32k_public_v1.jsonl`

- Confirmed stats so far:
  - `records=128`
  - `by_source={soar_public:128}`
  - `by_task={mcq:30,qa:30,niah:30,fwe:19,cwe:19}`
  - `TOK_P50=31679`
  - `TOK_P90=32768`
  - `TOK_MAX=32768`

- Preliminary judgment before bucket-count refinement:
  - the leading choice is still `64`, not `80/96`

- Why `64` is currently favored:
  - the token distribution is already heavily saturated near the `32K` cap, so each sample is expensive
  - this means `64 x 32K` already gives a large long-context calibration budget
  - without evidence that the extra `16K-32K` tail beyond the first `64` samples adds materially different coverage, `80/96` is more likely to add cost than signal

- What could still move the choice upward:
  - if the remaining bucket counts show that well beyond the first `64` samples there are still many additional unique `16K-32K` examples and very few `<16K` examples, then `80` becomes plausible
  - `96` should still be treated as unlikely unless the extra `32` samples are demonstrably long and diverse

- Current draft single-candidate command:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated32k64-public-v1 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl --local-calib-samples 64 --max-length 32768 --min-good-batches 48 --overwrite'`

- Finalization rule:
  - once `<16K / 16K-32K / >=32K` counts are available, either confirm this `64` command or promote it once to `80`
  - do not promote straight to `96` without unusually strong evidence from those counts

## 2026-04-09 Quant Mainline Execution Rule

- Until the current official eval / quant queue is released, do not start any new heavy PTQ run.
- Once the queue is released, the first NVFP4 candidate to launch is `mlp_weight_only + bfloat16 + 32K + 64`.

## 2026-04-09 Queue Update

- Quant queue logic has been updated:
  - after the current official eval finishes, run base `fast-eval` first
  - only after that, launch the first NVFP4 candidate:
    - `mlp_weight_only + bfloat16 + 32K + 64`

## 2026-04-09 Route Lock

- From this pass, do not infer any new remote state from local sandbox SSH attempts.
- `gptq_marlin` stays first in the waiting quant queue.
- `NVFP4 mlp_weight_only + bfloat16 + 32K + 64` remains the prepared fallback.
- Do not change route priority based on this pass.

## 2026-04-09 Active Quant Gate While Official Eval Runs

- Active control run:
  - remote tmux window: `stock-official-eval`
  - control run dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Quant-side execution rule until that run finishes:
  - no new heavy quantization job
  - no new quantized serve/smoke
  - keep `W4A16 + GPTQ + Marlin` as the queued primary route, but do not judge it against the 5-sample smoke
- Immediate lightweight step after the official eval ends:
  - verify the local GPTQ checkpoint shape before spending the next free GPU window

```bash
ssh rtx6000-2 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
p = Path("/root/autodl-tmp/models-gptq-w4a16-v19")
required = ["config.json", "quantize_config.json", "tokenizer_config.json"]
missing = [name for name in required if not (p / name).exists()]
weights = sorted(x.name for x in p.glob("*.safetensors"))
print({"path": str(p), "missing": missing, "num_safetensors": len(weights), "sample_weights": weights[:3]})
raise SystemExit(0 if not missing and weights else 1)
PY'
```

- Decision gate after that preflight:
  - if the checkpoint is structurally present and the official stock-base eval is not obviously invalid, launch the queued `gptq_marlin` smoke
  - otherwise, fix the checkpoint or control path first and keep NVFP4 in fallback position only

## 2026-04-09 Quant Hold While Official Stock Eval Runs

- Active quality gate:
  - official source-of-truth run is in progress at:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
  - launcher:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/run_official_stock_eval.sh`
- Quant policy during this run:
  - do not launch any new quantization-heavy experiment
  - do not rerank routes from `fast_test.sh`
  - keep `W4A16 + GPTQ + Marlin` as the next primary candidate, but hold the launch decision behind the official stock-base result
- Next quant action after the official eval finishes:
  - run the existing structural preflight on `/root/autodl-tmp/models-gptq-w4a16-v19`
  - if that preflight passes and the official stock-base eval is technically valid, launch the queued `gptq_marlin` smoke

## 2026-04-09 Post-Official-Eval Quant Gate

- Keep heavy GPU work paused until the official stock-base eval finishes.
- First lightweight preflight after that run ends:

```bash
ssh rtx6000-2 'RUN_DIR=$(cat /root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_latest.txt) && echo RUN_DIR=$RUN_DIR && rg -n "Average Score|Total Duration|Overall TPS|Request failed|ERROR|HTTP" "$RUN_DIR/eval_model.log" "$RUN_DIR/server.log" || true && tail -n 20 "$RUN_DIR/eval_model.log"'
```

- Treat the official stock-base eval as technically healthy only if:
  - `eval_model.log` contains `Average Score:` and `Total Duration:`
  - there are no obvious `Request failed`, `HTTP`, or `ERROR` lines indicating transport/runtime failure
  - the run looks like a completed full-quality pass rather than an early abort
- Healthy branch:
  - next candidate remains `W4A16 + GPTQ + Marlin`
  - keep the existing local GPTQ checkpoint as first path
  - run this lightweight checkpoint-structure preflight immediately before launch:

```bash
ssh rtx6000-2 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
p = Path("/root/autodl-tmp/models-gptq-w4a16-v19")
required = ["config.json", "quantize_config.json", "tokenizer_config.json"]
missing = [name for name in required if not (p / name).exists()]
weights = sorted(x.name for x in p.glob("*.safetensors"))
print({"path": str(p), "missing": missing, "num_safetensors": len(weights), "sample_weights": weights[:3]})
raise SystemExit(0 if not missing and weights else 1)
PY'
```

  - if that passes, exact next launch command is:

```bash
ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models-gptq-w4a16-v19 --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --quantization gptq_marlin --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_server.log 2>&1 &'
ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_eval.log'
```

- Unhealthy branch:
  - do not launch any quant candidate yet
  - next stronger stock-base quality-control check is the queued `mini_test.sh` control on the stable raw-model serve path
  - exact next launch command in that branch is:

```bash
ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_server.log 2>&1 & sleep 20 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=4 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 bash mini_test.sh | tee /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_mini.log'
```

## 2026-04-09 Quant Official Queue Ready

- Current official route priority remains:
  1. `official_gptq_marlin`
  2. `official_nvfp4_curated16k128`
- Queue state in this pass:
  - remote tmux window:
    - `codex-soar:quant-official-queue`
  - queue log:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/quant_official_queue.log`
  - behavior:
    - waits until the active stock-base official eval/server finishes
    - then runs the queued official full evals serially
- Candidate preflight confirmations from this pass:
  - GPTQ route:
    - `/root/autodl-tmp/models-gptq-w4a16-v19`
    - required files present
    - `2` safetensor shards present
  - NVFP4 fallback route:
    - `/root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128`
    - required files present
    - `2` safetensor shards present
- Current short decision template:
  - keep pressing `gptq_marlin` if:
    - it completes cleanly
    - and the official result is at least competitive with stock-base
    - and it clearly beats `NVFP4 curated16k-128`
  - fall back to `NVFP4 curated16k-128` if:
    - `gptq_marlin` fails technically
    - or its official result is clearly worse than both stock-base and NVFP4
  - allow a third candidate only if:
    - both first candidates run cleanly
    - but neither official result is good enough
  - bounded third-candidate rule:
    - use only `NVFP4 MLP-only curated16k-256`
    - do not open a wider search while the first two official comparisons remain unresolved

## 2026-04-09 Mid-Run Quant Refresh

- Official stock-base eval is still running; quant heavy work stays paused.
- This pass confirmed the next quant branch remains binary:
  - healthy official stock eval:
    - `W4A16 + GPTQ + Marlin`
  - weak or technically unhealthy official stock eval:
    - `stock-base mini control` first
- New small blocker captured:
  - remote host lacks `rg`
  - quant post-run scans and decision helpers must use `grep -E`
- Fresh quant delivery in this pass:
  - exact healthy-branch command delivered
  - exact weak-branch fallback command delivered
- Follow-up assigned in this pass:
  - quant worker must prepare a short operational rubric for when the official stock eval is healthy enough to proceed directly to `gptq_marlin`

## 2026-04-09 Quant Hold: Remote DNS Blocker

- Fresh quant delivery in this pass:
  - no quant route should launch while the stock official eval remains the single active heavy job
  - the next quant route is still:
    - `W4A16 + GPTQ + Marlin`
- Cheap preflight result from the orchestrator pass:
  - `ssh -G rtx6000-2` shows the alias still points to:
    - `connect.bjb1.seetacloud.com:50884`
  - local DNS lookup for that host currently fails with:
    - `gaierror(8, 'nodename nor servname provided, or not known')`
- Quant follow-up after this result:
  - quant stays paused until reachability recovers
  - first action after DNS recovery remains:
    - zero-GPU checkpoint preflight for `/root/autodl-tmp/models-gptq-w4a16-v19`
  - only after that, and only if the stock official eval is technically healthy, allow the queued `gptq_marlin` path to proceed
- Worker bookkeeping for this pass:
  - quant worker delivered:
    - yes
  - quant worker received a concrete follow-up:
    - yes

## 2026-04-09 Quant Hold: Official Eval Still Running At 50/150

- Fresh quant delivery in this pass:
  - while the stock-base official eval is still running, quant should do no more than a log-tail/status preflight
  - after the official eval finishes, quant should immediately run the shared `grep -E` plus tail summary scan before choosing between `gptq_marlin` and stock-base mini control
- Current official-eval state seen by the orchestrator:
  - `50/150`
  - continued `200 OK`
  - no newly observed `ERROR`
  - no newly observed `Request failed`
- Follow-up assigned in this pass:
  - keep the two-branch quant gate and exact next-command text current in:
    - `/Users/ql/cursor/openbmb/quantization-log.md`
    - `/Users/ql/cursor/openbmb/marlin-route-log.md`

## 2026-04-09 Quant Comparison Versus Original Weights

- Fresh quant delivery in this pass:
  - original / stock-base local smoke remains:
    - `28.00%`
  - best quant local smoke seen so far remains:
    - `36%`
  - more stable curated NVFP4 local smoke remains:
    - about `32%`
- Current interpretation:
  - local quant routes show at most a small smoke-level edge over original weights
  - do not treat that as a real route win yet because the official 150-sample long-context result is still pending
- Cheap preflight completed in this pass:
  - zero-GPU structure check for `/root/autodl-tmp/models-gptq-w4a16-v19` passed
  - required config files are present
  - `2` safetensors shards are present
- Follow-up assigned in this pass:
  - quant worker must treat the official stock-base result as the last real gate before launching `gptq_marlin`

## 2026-04-09 Quant Branches Ready, Still Waiting On Official Stock Result

- Fresh quant delivery in this pass:
  - healthy official stock result:
    - launch `gptq_marlin` smoke next using `/root/autodl-tmp/models-gptq-w4a16-v19`
  - weak or abnormal official stock result:
    - run `stock-base mini_test.sh` first instead of launching a quant route
- Current orchestrator read:
  - official stock eval is still running at about `71/150`
  - queue is still correctly blocked behind it
- Follow-up assigned in this pass:
  - keep the two post-official quant branches current
  - do not start GPU work before the official stock result lands

## 2026-04-09 Quant Hold Under DNS Blocker

- This pass did not launch any quant-side run.
- The single gating preflight for this pass was a direct SSH reachability check, and it failed on DNS resolution for:
  - `connect.bjb1.seetacloud.com`
- Fresh quant delivery in this pass:
  - blocker report from the replacement quant worker
  - recommended no-GPU narrowing step:
    - `ssh -G rtx6000-2 | sed -n '1,20p'`
  - interpretation:
    - keep treating the blocker as local reachability, not checkpoint completeness
- Follow-up assigned in this pass:
  - quant side stays paused until DNS/reachability recovers or the official stock result becomes readable again

## 2026-04-09 Quant Third Possibility, Kept Bounded

- Fresh quant delivery in this pass:
  - third possibility:
    - reuse the existing `NVFP4` stable line as a submission-safe fallback
    - prefer existing artifacts such as:
      - `/root/autodl-tmp/models-nvfp4-mlp-only-local128`
      - current `nvfp4-mlp-only-draft`
- Trigger condition:
  - only if the main post-official branches do not look good enough
  - do not expand into a fresh quantization search
- Follow-up assigned in this pass:
  - keep this third possibility bounded as fallback-only

## 2026-04-09 Calibration Pressure Shift: Not More Copies Of 128, But Less 16K Saturation

- New hard facts from this pass:
  - current public set on disk is:
    - `0-4K: 30`
    - `16K-32K: 40`
    - `32K-128K: 80`
    - max prompt length:
      - `127738`
  - task-by-bucket shape:
    - `mcq`:
      - `30 x 0-4K`
    - `qa / niah / fwe / cwe`:
      - each `10 x 16K-32K`
      - each `20 x 32K-128K`
  - current active curated calibration file:
    - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v1.jsonl`
    - `128` JSON records
    - `128` unique texts
    - token stats:
      - `p50=16384`
      - `p90=16384`
      - `max=16385`
      - `>=16K: 67`
      - `>=8K: 76`
- Immediate interpretation:
  - another `local_calib_samples=256` run on the same `16K v1` file is not a real calibration change
  - the stronger next lever is:
    - either shrink quantization scope
    - or reduce the share of samples that are clipped at `16K`
- Fresh quant delivery in this pass:
  - best small-scope NVFP4 candidate remains:
    - `mlp_weight_only + bfloat16 + 16K + 128`
  - why:
    - current `bf16 + 16K + 128` is already technically healthy
    - `mlp_weight_only` removes MLP activation quantization pressure while keeping a small edit radius
    - it is a safer first correction than "same file, more copies"
- Remote prep started in this pass:
  - first `32K + pg19` build idea immediately hit a preparation blocker:
    - remote HF dataset access was attempted without:
      - `source /etc/network_turbo`
      - `HF_ENDPOINT=https://hf-mirror.com`
    - practical lesson:
      - keep the next calibration branch network-light unless we explicitly need outside corpora
  - follow-up prep fix in the same pass:
    - local builder now skips open-source iterators when quota is `0`
    - remote launch path now explicitly enables:
      - `source /etc/network_turbo`
      - `HF_ENDPOINT=https://hf-mirror.com`
  - current active remote calibration build:
    - tmux window:
      - `codex-soar:calib32k-public`
    - output:
      - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl`
    - build log:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/build_calibration_curated_32k_public_v1.log`
    - config:
      - `public_quotas=mcq:30,qa:30,niah:30,fwe:19,cwe:19`
      - `open_quotas=cnn_dailymail:0,wikitext:0`
      - `max_sample_tokens=32768`
      - `head=4096`
      - `middle=4096`
      - `tail=24576`
- Current next-candidate ranking:
  1. `NVFP4 mlp_weight_only + bf16 + 16K + 128`
  2. `W4A16 + GPTQ + Marlin` official full eval already queued as the main challenger
  3. only after the new `32K` calibration file is built and inspected:
     `NVFP4 mlp_weight_only + bf16 + 32K + reduced sample count`
- Guardrail:
  - do not jump straight to `32K + 128` on the eval machine
  - first inspect the new `32K` file's actual token distribution and then decide whether the safer dev-only candidate should be `64`, `80`, or `96` samples

## 2026-04-09 Networkless 32K Trigger Rule

- Fresh quant delivery in this pass:
  - if the new `32K` calibration file must be built from local public data only, the best next NVFP4 dev candidate becomes:
    - `mlp_weight_only + bfloat16 + 32K + 64 samples`
- Why this is the current winner:
  - it preserves the strong new tail-heavy `32K` calibration change
  - it avoids immediately paying the full `32K x 128` single-GPU PTQ bill
  - it is still a smaller, cleaner step than opening multiple `32K` sample-count variants
- Trigger rule:
  - wait for:
    - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl`
  - then inspect only:
    - record count
    - unique texts
    - `TOK_P50`
    - `TOK_P90`
    - `TOK_MAX`
    - share of samples in `16K-32K`
  - only after that decide:
    - keep `64`
    - or raise to `80/96`
  - do not queue multiple `32K` sample-count variants before that inspection

## 2026-04-09 32K Public-Only Stats And Provisional Sample Count

- The networkless `32K` calibration file is now built:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl`
- Measured stats:
  - `records=128`
  - `unique_texts=128`
  - `by_task={mcq:30, qa:30, niah:30, fwe:19, cwe:19}`
  - `by_bucket={0-4K:30, 16K-32K:32, 32K-128K:66}`
  - `TOK_P50=31655`
  - `TOK_P90=32769`
  - `TOK_MAX=32769`
  - `TOK_LT16K=30`
  - `TOK_16K_32K=43`
  - `TOK_GE32K=55`
- Current read:
  - this file is clearly long-context heavy enough
  - the dominant cost/risk is now the per-sample length, not lack of long-context coverage
  - because of that, the provisional winner remains:
    - `mlp_weight_only + bfloat16 + 32K + 64 samples`
  - `80/96` should be considered only if we later decide the first `32K x 64` PTQ run is too noisy but still technically stable
- Launch helper prepared in this pass:
  - `/Users/ql/cursor/openbmb/scripts/run_nvfp4_mlp_weight_only_32k64_remote.sh`
  - target export:
    - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-curated32k64-public-v1`
  - calibration file:
    - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl`

## 2026-04-09 Queue Updated To New Main Candidate

- Fresh quant follow-up in this pass:
  - the waiting `quant-v2-queue` is now aligned with the current main plan
  - after the stock official eval finishes, it will:
    1. run raw-model `fast-eval`
    2. launch the main quant candidate:
       - `NVFP4 + mlp_weight_only + bf16 + 32K + 64 samples`
    3. run the same `fast-eval` on the quantized result
    4. run broader local `mini` only if the fast proxy clears the basic gate
- Updated queue files:
  - local:
    - `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
  - remote deployed copy:
    - `/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/run_quant_fast_queue.sh`
- Execution rule remains:
  - until the current stock official eval / server is fully released, do not start any new heavy PTQ run manually

## 2026-04-09 Quant Prep Under DNS Blocker

- Direct remote quant follow-up remained blocked in this pass by local DNS failure for:
  - `connect.bjb1.seetacloud.com`
- Quant-side useful work completed anyway:
  - calibration builder gained:
    - configurable quotas
    - explicit `bucket` metadata
    - optional `pg19` source for longer open-text calibration
- Implication:
  - once reachability returns, the next calibration revision can be built without reopening the script design work

## 2026-04-09 Quant Focus Narrowed

- Fresh quant delivery in this pass produced the current priority problem list:
  1. official stock baseline is not finished yet
  2. `smoke` is too weak to settle quality
  3. any residual `rg`-based helper is a false blocker on this remote host
  4. `gptq_marlin` still needs one controlled live validation
  5. the current GPTQ artifact is structurally ready but quality-unproven
- Plan constraint after this pass:
  - keep the quant plan narrowed to:
    - `gptq_marlin`
    - existing `NVFP4` fallback
  - do not open a fresh quantization search branch
- Follow-up assigned in this pass:
  - keep the `rg` blocker tracked in quant-side commands and logs

## 2026-04-09 Calibration V2

- Built a new bounded calibration set:
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v2.jsonl`
- Current `v2` task mix:
  - `30 qa`
  - `30 niah`
  - `20 mcq`
  - `48 open_text`
  - `0 fwe`
  - `0 cwe`
- Current source mix:
  - `soar_public`
  - `cnn_dailymail`
  - `wikitext`
  - `merged_semantic_v2`
- Reason for the change:
  - align more closely with the champion guidance to reduce low-semantic random tasks in calibration
  - keep the search bounded and reuse existing local data instead of opening a new calibration-data hunt
- Default quant entrypoints updated to prefer `v2`:
  - `/Users/ql/cursor/openbmb/scripts/quantize_modelopt_nvfp4_conservative.py`
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py`
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/prepare_model.sh`
- Next quant command once the GPU window is free:
  - `python3 /root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-v2 --cfg mlp_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v2.jsonl --local-calib-samples 128 --max-length 16384 --min-good-batches 96 --overwrite`

## 2026-04-09 V2 Queue Active

- The v2 quant route is now armed in remote tmux:
  - `codex-soar:quant-v2-queue`
- Active queue script:
  - `/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/run_quant_fast_queue.sh`
- Queue logic:
  - wait for the stock official eval and stock server to finish
  - run v2 NVFP4 quantization
  - run local `smoke`
  - only if `smoke` is technically healthy, continue to local `mini`
- Current blocker is therefore:
  - GPU occupancy by the still-running stock official eval
  - not missing calibration preparation and not missing draft files

## 2026-04-09 First Quant Fallback After V2

- Current expected first failure signal once `v2` starts:
  - `nan amax`
  - or too few `good_batches`
- First minimal fallback if that happens:
  - keep:
    - `calibration_curated_16k_v2.jsonl`
    - `128` local calibration samples
    - `max-length 16384`
  - change only:
    - `--cfg mlp_only` -> `--cfg mlp_weight_only`
- Reason:
  - this is the smallest scope reduction that still reuses the new calibration work and directly targets stability

## 2026-04-09 Precision Gate Reconfirmed

- Quantization remains the active priority.
- Speed work stays paused until a new quantized route is:
  - technically healthy on the same harness
  - and no worse than original weights by more than `1-2` absolute points on `official` or at least `stock-base mini`
- Current route lock remains:
  - `gptq_marlin` mainline
  - existing `NVFP4` fallback

## 2026-04-09 Quant Acceptance Gate And Local Runnability Check

- This pass did not launch a quant run.
- Reason:
  - the official stock-base full eval is still the active long run from the last confirmed remote state
  - fresh remote access from this sandbox was blocked, so the direct next gate was local quant-runnability, not another SSH-triggered run
- Local readiness check completed:
  - `/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v2.jsonl`
    - exists
    - `128` rows
    - task mix:
      - `qa 30`
      - `niah 30`
      - `mcq 20`
      - `open_text 48`
  - both quant scripts and draft `prepare_model.sh` point at:
    - `calibration_curated_16k_v2.jsonl`
  - syntax checks passed:
    - local quant script
    - draft quant script
    - draft `prepare_model.sh`
- Fresh quant delivery in this pass:
  - acceptance rule:
    - `close_enough = clean run AND no material pathology AND quant_avg >= 0.90 * stock_base_avg`
- Fresh quant follow-up delivery in this pass:
  - route-selection rule after the official result:
    - if `gptq_marlin` is clean and meets the `0.90 * stock` bar, run `gptq_marlin` first
    - else promote `NVFP4 curated16k v2`
- Worker delivery in this pass:
  - yes
- Worker follow-up assigned in this pass:
  - yes

## 2026-04-09 Quant Rediscovery Rule After Lost Tmux State

- Fresh quant delivery in this pass:
  - keep `gptq_marlin` first
  - keep the immediate post-official gate as the local GPTQ checkpoint structure preflight
- Cheap preflight blocker in this pass:
  - remote `tmux` windows expected for:
    - `stock-official-eval`
    - `quant-v2-queue`
  - were both missing
- Therefore:
  - do not infer official completion or queue advancement from this pass
  - do not launch Marlin or NVFP4 from this pass
- Fresh quant follow-up delivery in this pass:
  - first non-GPU rediscovery command:
    - `ssh rtx6000-2 'cd /root/autodl-tmp/SOAR-Toolkit/test_results && ls -td official_stock_base_* gptq_marlin_* nvfp4_* candidate_benches/* 2>/dev/null | head -n 20'`
  - quant-side reason:
    - re-establish authoritative run directories before any route is resumed

## 2026-04-09 Quant Hold Rule Under Broken Fast-Proxy Gate

- Fresh quant delivery in this pass:
  - stop using the current `fast` result as a quant ranking gate
- Reason:
  - stock-base `fast` already showed:
    - short `mcq`
    - `req_out=2767`
    - `finish=length`
  - therefore the proxy harness was contaminating route judgment
- Quant-side implication:
  - neither `Marlin` nor `NVFP4 32K64` should be promoted or eliminated until the proxy budget path is fixed and rerun

## 2026-04-09 Route Update: Marlin Promoted, NVFP4 32K64 Remains Prepared Fallback

- Fresh quant delivery in this pass:
  - current precision-first lead route:
    - `W4A16 + GPTQ + Marlin (local-GPTQ-first)`
  - prepared fallback remains:
    - `NVFP4 mlp_weight_only + bf16 + 32K + 64`
- Reason for the promotion:
  - this is the champion-aligned route
  - remote already has a structurally complete local GPTQ checkpoint at:
    - `/root/autodl-tmp/models-gptq-w4a16-v19`
  - therefore this route can be gated without launching another fresh heavy PTQ first
- Concrete first handoff from the quant side:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && export SOAR_GPTQ_LOCAL_PATH=/root/autodl-tmp/models-gptq-w4a16-v19 && bash /root/autodl-tmp/codex-drafts/w4a16-marlin-draft/prepare_model.sh --input /root/autodl-tmp/models --output /root/autodl-tmp/models-gptq-marlin-local-prepared'`
- Fresh quant-side risk report in this pass:
  - main submission risk was:
    - `prepare_model/preprocess` depended too much on an external `SOAR_GPTQ_LOCAL_PATH`
  - implication:
    - if that path were absent on the target machine, the route would fail before serving
- Fix completed in this pass:
  - updated `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/preprocess_model.py`
    - it now auto-detects a small ordered list of known local GPTQ candidate paths before failing
    - it now prints the searched candidate list on failure
  - updated `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/prepare_model.sh`
    - it now enables `/etc/network_turbo` when present
    - it now sets `HF_ENDPOINT=https://hf-mirror.com` before any fallback download path
  - synced both files to:
    - `/root/autodl-tmp/codex-drafts/w4a16-marlin-draft/`
  - verified:
    - local `py_compile` / `bash -n` OK
    - remote `py_compile` / `bash -n` OK
- Route order after this pass:
  1. `W4A16 + GPTQ + Marlin (local-GPTQ-first)`
  2. `NVFP4 mlp_weight_only + bf16 + 32K + 64`

## 2026-04-09 Quant Decision Reconfirmed After Stock Official Exit

- New stock state in this pass:
  - the stock official process ended
  - but the expected output directory:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260409_034351`
    - is empty
  - therefore do not treat that run as a settled precision anchor
- Fresh quant follow-up delivery in this pass:
  - if the current `base fast` control is technically healthy, the next quant route should still be:
    - `W4A16 + GPTQ + Marlin (local-GPTQ-first)`
  - one-line reason:
    - this remains the champion-aligned highest-priority precision route, while `NVFP4 32K64` stays the prepared fallback
- New active control gate before any quant route:
  - `stock-base-fast-eval`
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_045057_stock-base-fast-eval`
- Current route order remains:
  1. `W4A16 + GPTQ + Marlin (local-GPTQ-first)`
  2. `NVFP4 mlp_weight_only + bf16 + 32K + 64`

## 2026-04-09 Marlin Still Next After Proxy Repair

- New stock-base proxy evidence in this pass:
  - the first uncapped proxy control proved the harness budget was wrong
  - after the task-aware cap fix, the capped proxy control became usable
  - first observed capped rows:
    - `mcq` now requests `64`, not `2767`
    - there is no server error in the observed prefix
    - the stock base quality still looks weak, but the proxy itself is no longer obviously misleading
- Fresh quant follow-up delivery in this pass:
  - once the capped proxy control is done, the next single quant step should be:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && export SOAR_GPTQ_LOCAL_PATH=/root/autodl-tmp/models-gptq-w4a16-v19 && bash /root/autodl-tmp/codex-drafts/w4a16-marlin-draft/prepare_model.sh --input /root/autodl-tmp/models --output /root/autodl-tmp/models-gptq-marlin-local-prepared'`
- Route order remains:
  1. `W4A16 + GPTQ + Marlin (local-GPTQ-first)`
  2. `NVFP4 mlp_weight_only + bf16 + 32K + 64`

## 2026-04-09 Four-Signal Decision Order

- Only compare these four signals:
  1. `stock official`
  2. `base fast`
  3. `quant fast`
  4. `quant mini`

- Decision order:
  - read `stock official` first:
    - this is the anchor for what raw MiniCPM-SALA can actually do
  - then read `base fast`:
    - this tells us whether the local proxy is directionally trustworthy against the official anchor
  - then read `quant fast`:
    - this is the first cheap quant gate
  - only if `quant fast` is technically healthy, read `quant mini`

- Elimination rules:
  - drop the quant route immediately if `quant fast` has obvious technical failure:
    - request failures
    - empty outputs
    - broad `finish_reason=length` pathology
  - drop the quant route if `quant fast` is clearly worse than `base fast`
  - do not promote the quant route on `fast` alone
  - promote the quant route only if `quant mini` stays directionally consistent with `quant fast` and does not materially undercut the `stock official` anchor

- Practical shortcut:
  - `stock official` defines the ceiling
  - `base fast` validates the local proxy
  - `quant fast` decides whether the route survives
  - `quant mini` decides whether the surviving route is worth keeping

## 2026-04-09 Single Fallback Rule

- If `32K64` fails clearly at `fast`, use exactly one conservative fallback next:
  - `mlp_weight_only + bfloat16 + 16K + 128`
- Execution rule:
  - if `quant fast` for `mlp_weight_only + bfloat16 + 32K + 64` is clearly bad, do not search further; launch only `mlp_weight_only + bfloat16 + 16K + 128` as the sole fallback candidate
- Exact fallback command:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated16k-128 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v1.jsonl --local-calib-samples 128 --max-length 16384 --min-good-batches 64 --overwrite'`

## 2026-04-09 Quant Post-Run Extraction Snippet

- Use this right after `fast` or `mini` finishes:
  - `LOG=<log_path>; rg -n "avg_score=|pass=|part=|fail=|empty|finish_reason=length" "$LOG" -S`
- If you want both logs at once:
  - `for LOG in <fast_log> <mini_log>; do echo "--- $LOG ---"; rg -n "avg_score=|pass=|part=|fail=|empty|finish_reason=length" "$LOG" -S; done`

## 2026-04-09 Route Lock Reaffirmed Under SSH Blocker

- Fresh quant delivery in this pass:
  - keep `gptq_marlin` first
  - keep `NVFP4 mlp_weight_only + bf16 + 32K + 64` as prepared fallback only
- Reason recorded in this pass:
  - `gptq_marlin` remains the champion-aligned highest-upside route
  - the current `NVFP4` line is prepared and useful, but not promoted ahead of it from this pass
- Cheap preflight blocker in this pass:
  - a fresh direct remote SSH check failed with:
    - `Operation not permitted`
- Therefore:
  - make no new remote-state assumption from this pass
  - do not change route order because of this pass alone

## 2026-04-09 Quant Gate Reconfirmed After Fresh Worker Follow-Up

- Fresh quant delivery in this pass:
  - exact next `gptq_marlin` server boot command remains the first quant handoff after the stock official eval
- Fresh follow-up delivery in this pass:
  - exact next quant-eval command:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_fast_eval.log'`
  - keep `gptq_marlin` first only if:
    - the run is clean
    - there is no obvious pathology:
      - `Request failed`
      - `ERROR`
      - broad `finish_reason=length`
    - and `avg_score >= base fast`
  - else:
    - hand off immediately to `NVFP4 mlp_weight_only + bf16 + 32K + 64`
- New blocker in this pass:
  - fresh direct-IP remote reachability check from this sandbox failed with:
    - `Operation not permitted`
- Therefore:
  - route order stays:
    1. `gptq_marlin`
    2. `NVFP4 32K64` fallback
  - no new remote quant run was started from this pass
- Worker delivery in this pass:
  - yes
- Worker follow-up assigned in this pass:
  - yes

## 2026-04-09 Quant Follow-Up After Runnable Fast-Proxy Runner Preflight

- Fresh quant delivery in this pass:
  - if the patched runner preflight passes, the next quant-side command remains:
    - `python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py --label gptq-marlin-fast-v1 --model-path /root/autodl-tmp/models-gptq-marlin-local-prepared --quantization gptq_marlin --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --profile fast-eval --disable-radix-cache --disable-cuda-graph --dense-as-sparse`
- Cheap preflight completed in this pass:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - result:
    - passed
- Non-blocking side observation:
  - an attempted `py_compile` on `soar-patches/mini_test.sh` failed only because it is a shell script
  - do not treat that as a quant blocker
- Route rule unchanged:
  - keep `gptq_marlin` first only after its first live fast run is clean and directionally healthy
  - otherwise hand off to `NVFP4 mlp_weight_only + bf16 + 32K + 64`
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Quant Hold Rule Under Sandbox-Blocked Marlin Metadata Check

- Fresh quant delivery in this pass:
  - do not modify `/root/autodl-tmp/models-gptq-marlin-local-prepared`
  - do not demote to `NVFP4` from this pass alone
  - next quant move must be one read-only metadata check once trusted SSH is available again
- Cheap preflight attempted in this pass:
  - inspect prepared `Marlin` metadata and weight-map naming to decide whether a metadata remap is necessary
- Result:
  - blocked at the local sandbox layer before execution:
    - `ssh: connect to host 106.120.183.117 port 50884: Operation not permitted`
- Therefore:
  - this pass gathered no trustworthy new remote metadata
  - do not perform a destructive metadata edit from this pass
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Analysis Of Latest Successful NVFP4 Submission

- Observed official result from the latest successful NVFP4 submission:
  - `acc_ori = 11.47`
  - `acc = 14.33`
  - `benchmark_duration.S1 = 1062.89`
  - `benchmark_duration.S8 = 813.59`
  - `benchmark_duration.Smax = 1324.65`
  - `final_score = 0.0`

- First conclusion:
  - this route is not merely "slightly worse"
  - it is a failed competition route in its current form

- Why `final_score` became `0.0`:
  - official correctness scoring in the toolkit normalizes against an `80`-point base model reference
  - `11.47 / 80 * 100 ≈ 14.34`, which matches the returned `acc = 14.33`
  - that means the submission preserved only about `14%` of the required correctness baseline
  - by the champion scoring notes, such a low correctness coefficient effectively zeroes the final score

- Likely accuracy failure sources, ordered by confidence:

1. `ModelOpt NVFP4` is a poor match for the current MiniCPM-SALA stack.
   - existing local notes already flagged that MiniCPM-SALA custom sparse attention is not fully supported by the `modelopt` NVFP4 path
   - this is consistent with a route that can be made runnable but still collapses in accuracy

2. The submission route can silently fall back to a much weaker quantization attempt.
   - current `prepare_model.sh` tries:
     - `16K bf16`
     - then `8K bf16`
     - then `2048 fp16`
   - so a "successful" submission does not prove the intended `16K` route was actually what got served
   - if the official machine fell through to the `2048` fallback, long-context correctness would be expected to collapse

3. The calibration distribution is still mismatched to the real evaluation regime.
   - current packaged route is centered on `16K` and `128` samples
   - the official public set is dominated by `16K-32K` and especially `32K-128K`
   - the submission also removed `fwe/cwe` from the curated `v2` mix, which may reduce direct task coverage for two of the five public task families

4. The local smoke numbers were misleadingly optimistic.
   - the preserved local comparison `base=28%`, `NVFP4 smoke=36%` was only a short smoke gate
   - that signal was never strong enough to justify trusting the route on the official set

- Likely speed failure sources, ordered by confidence:

1. This route does not really hit the main decode bottleneck.
   - the submission quantizes via `modelopt`
   - but the route is still effectively `MLP-only` / partial weight compression, not the champion-style `W4A16 + GPTQ + Marlin`
   - for MiniCPM-SALA long-context inference, KV/state/sparse-attention behavior remains a major cost, so shrinking only part of the weights does not guarantee speedup

2. `modelopt` runtime likely pays generic dequant / fallback overhead without enough kernel win.
   - the serve path is just:
     - `--quantization modelopt --dtype float16`
   - there is no evidence this route is hitting a truly optimized MiniCPM-specific fast path comparable to Marlin
   - a compressed checkpoint can therefore be smaller on disk but still slower at runtime

3. The submission kept conservative stability flags instead of speed-oriented ones.
   - current server args remain:
     - `--disable-radix-cache`
     - `--disable-cuda-graph`
     - `--chunked-prefill-size 8192`
     - `--dense-as-sparse`
   - these are reasonable for stability, but they do not create a speed win by themselves

4. Weight quantization does not solve long-context memory pressure by itself.
   - even if model weights shrink, long-context evaluation is still dominated by runtime cache/state
   - this explains the user-visible feeling of "又菜又满": correctness collapsed, while runtime still behaved like a heavy long-context model

- Important comparison correction:
  - our local notes do **not** support the claim that the earlier selective RTN path was already a good final route
  - the preserved RTN notes say:
    - the selective RTN prototype could quantize and load
    - but local ultra-short eval still collapsed to `0.00%`
  - the earlier official score around `37.36` belongs to the `v3-fusion-flashinfer` line, not to the current NVFP4 draft

- Practical decision:
  - treat this latest NVFP4 submission as a strong negative result
  - do not invest further in this exact `modelopt NVFP4` submission scaffold as a primary route
  - if NVFP4 is revisited, it should be treated as a new route with a different kernel / support hypothesis, not as a small tweak of this draft
  - primary quant focus should stay on `W4A16 + GPTQ + Marlin`

## 2026-04-09 Quant Hold Rule After Medium-Eval Status Check Failed At Sandbox Layer

- Fresh quant delivery in this pass:
  - after `medium-eval`, the next quant move is still exactly one read-only metadata/schema inspection on:
    - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
  - until that happens:
    - do not modify checkpoint metadata
    - do not relaunch `gptq_marlin`
    - do not demote to `NVFP4` from this pass alone
- Cheap preflight attempted in this pass:
  - one direct remote status check for the active `medium-eval`
- Result:
  - blocked at the sandbox layer:
    - `ssh: connect to host 106.120.183.117 port 50884: Operation not permitted`
- Therefore:
  - no new quant route decision was made from this pass
  - quant stays on no-launch hold until trusted SSH returns
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Quant Delivery After Local Medium-Eval Readiness Patch

- Fresh quant delivery in this pass:
  - yes
- Quant handoff:
  - once `medium-eval` is trustworthy and trusted SSH returns, the next quant move remains exactly one read-only metadata/schema inspection on:
    - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
  - inspect only:
    - `config.json`
    - `quantize_config.json`
    - `model.safetensors.index.json`
  - do not relaunch `gptq_marlin` yet
  - do not demote to `NVFP4` from this pass
- Why no route change was made:
  - this pass consumed its single cheap preflight on the local medium-eval launcher readiness fix
  - there was no trusted remote read in this pass
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Quant Baseline Updated From Stock Medium-Eval

- Trusted control baseline now exists on original weights:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/medium_eval_stock_base_20260409_104835`
  - output:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260409_104905`
- Baseline quality:
  - `Original Accuracy: 50.40%`
  - `Normalized Accuracy: 63.0%`
- What this means for quant work:
  - route ranking must no longer rely primarily on `smoke`
  - a quant route that looks acceptable on `smoke` but is far below this medium-eval anchor should be treated as non-competitive
  - the latest weak `NVFP4` official result is now even more clearly a route-level failure relative to the real stock-base baseline
- Important caveat:
  - this medium-eval duration is inflated by runaway outputs and should not be used as the speed target
  - it is a correctness anchor first

## 2026-04-09 New Fast-Eval Anchor And Fresh NVFP4 Queue

- Old fast/proxy artifacts removed on remote:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/autorun_v6_fast_test.log`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/autorun_v7_fast_test.log`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_045057_stock-base-fast-eval`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_045846_stock-base-fast-eval-capped`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_101035_gptq-marlin-fast-eval-capped`
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`

- Replacement fast-eval dataset:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.meta.json`
- Replacement fast-eval design:
  - derived directly from `medium_eval`
  - `20` rows total
  - `4` rows per task
  - length mix:
    - `0-4K`: `4`
    - `16K-32K`: `4`
    - `32K-128K`: `12`
- Interpretation:
  - this is now the primary quick correctness gate
  - `smoke` remains only a smoke signal

- Fresh quant route launched:
  - route:
    - `NVFP4 mlp_weight_only + bfloat16 + 32K + 64`
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/quant_fastmed_nvfp4_mlp_weight_only_32k64_20260409_171614`
  - export target:
    - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-32k64-fastmed-v1`
  - execution path:
    1. quantize
    2. `fast eval`
    3. `medium eval`

- Fresh technical warning to remember:
  - `modelopt` could not create a quantized attention class for:
    - `MiniCPMInfLLMv2Attention`
    - `LightningAttention`
  - implication:
    - this route should be treated as a partial-coverage quant path, not a true full-stack attention-quantized route
    - if speed gains disappoint, this warning is one of the first explanations to check

## 2026-04-09 NVFP4 32K64 BF16 Route Classified As Runtime-Invalid

- Route:
  - `NVFP4 mlp_weight_only + bfloat16 + 32K + 64`
- Artifacts:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/quant_fastmed_nvfp4_mlp_weight_only_32k64_20260409_171614`
  - export:
    - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-32k64-fastmed-v1`
  - manifest:
    - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-32k64-fastmed-v1/codex_quant_manifest.json`
  - quant config:
    - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-32k64-fastmed-v1/hf_quant_config.json`
- Quantization stage:
  - succeeded
  - `good_batches=64`
  - `skipped_batches=0`
- Runtime stage:
  - failed on first long live batch
  - server warning:
    - `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0`
  - runtime cast:
    - `Casting torch.bfloat16 to torch.float16`
  - fatal error:
    - `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
- Result classification:
  - this route is not "low accuracy"
  - this route is `runtime-invalid`
  - `0.00%` fast and medium scores must not be interpreted as a true accuracy number

## 2026-04-09 Fast-Eval Gate Tightened To Capped Medium-Derived Subset

- Previous issue:
  - `fast-eval v2` preserved original large `completion_tokens`
  - that allowed runaway generations and made the supposed fast gate too slow
- New fast gate definition:
  - keep the medium-derived `20`-sample subset
  - keep `4` samples per task
  - cap `completion_tokens` by task:
    - `mcq=64`
    - `qa=128`
    - `niah=128`
    - `fwe=256`
    - `cwe=256`
- Rationale:
  - preserve task/length distribution signal
  - avoid letting `2` runaway rows dominate the gate
  - make fast-vs-medium quant comparisons practical again

## 2026-04-09 Active Retry: FP16 Export + DeepGemm Disabled

- Current active retry in tmux:
  - `codex-soar:quant-fastmed-fp16`
- Route:
  - `NVFP4 mlp_weight_only + float16 + 32K + 64`
- Export target:
  - `/root/autodl-tmp/models-nvfp4-mlp-weight-only-32k64-fp16-dg0`
- What changed relative to the failed BF16 route:
  - export dtype:
    - `bfloat16 -> float16`
  - serve dtype:
    - stays `float16`
  - runtime:
    - `DeepGemm disabled`
- What stayed fixed:
  - same quant cfg:
    - `mlp_weight_only`
  - same calibration file:
    - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl`
  - same calibration sample count:
    - `64`
  - same max length:
    - `32768`
- Decision rule:
  - if this retry still crashes or collapses far below stock medium baseline, `NVFP4` should be deprioritized behind `W4A16 + GPTQ/Marlin`

## 2026-04-09 Quant Gate Naming Normalized To `fast`

- Canonical quick-gate name is now just:
  - `fast`
- Canonical dataset path is now:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
- Canonical helper scripts are now:
  - `/Users/ql/cursor/openbmb/scripts/run_fast_eval_remote.sh`
  - `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
- Current original-weight control using the repaired `fast` gate:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_175739_stock-base-fast-eval`
- Status at log time:
  - still running, so no new quant-vs-stock score comparison yet
- Current quant interpretation remains:
  - `NVFP4 bf16 32K64`:
    - runtime-invalid
  - `NVFP4 fp16 32K64`:
    - calibration-invalid
  - next meaningful quant route should be judged against the repaired stock `fast` baseline only after that baseline finishes

## 2026-04-09 GPTQ Calibration Policy Updated

- Explicitly deprecated for active quant work:
  - `calibration_data_merged.jsonl`
- Reason:
  - poor task/length quality for the new GPTQ route
  - user explicitly rejected it
- Active GPTQ calibration file:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_v1.jsonl`
- Active GPTQ calibration builder:
  - `/Users/ql/cursor/openbmb/scripts/build_gptq_calibration.py`
- Composition:
  - `48` public-task rows
  - `16` synthetic ultra-long rows
- Current distribution:
  - by source:
    - `soar_public: 48`
    - `synthetic_ultralong: 16`
  - by bucket:
    - `0-4K: 12`
    - `16K-32K: 12`
    - `32K-128K: 33`
    - `128K-160K: 7`
- Important note:
  - the `128K-160K` coverage is synthetic because the public set itself does not contain genuine `128K-160K` prompts

## 2026-04-09 HF Download Policy For Quantization

- `source /etc/network_turbo` is helpful but insufficient.
- Confirmed remote helper file:
  - `/etc/network_turbo`
- It may improve:
  - GitHub / HuggingFace reachability
  - transfer speed
- It does not reliably fix:
  - old `datasets` compatibility problems
  - dataset-script deprecations
  - mirror-side `500` failures
- Therefore:
  - do not hinge the GPTQ calibration path on HF datasets
  - prefer public-set-derived calibration and synthetic long-context construction

## 2026-04-09 GPTQ W4A16 Mainline Blocker

- Active sanity output path:
  - `/root/autodl-tmp/models-gptq-w4a16-sanity16`
- Latest live attempt:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_w4a16_sanity16_retry3_20260409_184250.log`
- Progress:
  - loader compat issue bypassed
  - layer 0 quantization began and completed
  - some modules used `rtn failsafe` due Hessian recovery failure
- Current hard blocker:
  - layer 1 fails because:
    - `self_attn.o_gate` exists in early MiniCPM-SALA attention layers
    - later `LightningAttention` layers do not expose `o_gate`
  - `gptqmodel` still expects one static module tree across all layers
- Current conclusion:
  - next fix must be a SALA-specific GPTQ model definition or a dynamic per-layer module mapping
  - not more calibration tweaking yet

## 2026-04-09 GPTQ W4A16 Support Registration Fix

- Root cause refinement:
  - previous script only changed `MODEL_MAP`
  - `gptqmodel` still treated `minicpm_sala` as unsupported because:
    - `SUPPORTED_MODELS` did not include it
  - that forced a fallback to:
    - `BaseQModel`
    - `Module Tree AutoCompat`
  - AutoCompat copied layer-0 attention structure and reintroduced:
    - `self_attn.o_gate`
- Active fix now in:
  - `/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py`
- What changed:
  - created a dedicated `MiniCPMSALAGPTQ` subclass
  - registered:
    - `MODEL_MAP["minicpm_sala"] = MiniCPMSALAGPTQ`
  - appended:
    - `SUPPORTED_MODELS.append("minicpm_sala")`
  - set:
    - `layer_modules_strict = False`
  - attention module tree now uses only common linears:
    - `q_proj`
    - `k_proj`
    - `v_proj`
    - `o_proj`
- Current live run after the fix:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_w4a16_sanity16_20260409_185650.log`
- Live signal:
  - `AutoCompat` log line disappeared
  - run progressed into:
    - calibration input capture
- Current interpretation:
  - route is no longer blocked at model-definition registration
  - next remaining question is whether the run can finish and save a checkpoint cleanly

## 2026-04-09 `fast` Gate Reduced Without Renaming

- Canonical path is unchanged:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
- Canonical name is unchanged:
  - `fast`
- Selection rule updated:
  - keep exactly one row per available task/length-bucket pair from medium-eval
- Current realized dataset:
  - `9` rows total
  - buckets:
    - `0-4K: 1`
    - `16K-32K: 4`
    - `32K-128K: 4`
- This is now the active quick gate for the GPTQ route.

## 2026-04-09 GPTQ Route Not Ready For `fast` Yet

- Active run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_w4a16_sanity16_20260409_185650.log`
- Current checkpoint status:
  - `quantize_config.json` not present yet under:
    - `/root/autodl-tmp/models-gptq-w4a16-sanity16`
- Therefore:
  - no new-weight `fast` run has been launched yet
- Current live behavior:
  - quantization is progressing through layers
  - many modules are using damp-recovery and some fall back to RTN failsafe
- Interpretation:
  - structural blocker is fixed
  - numerical stability / quality is now the main remaining risk

## 2026-04-09 GPTQ Live Failure Analysis And Tooling Read

- Latest live status:
  - run is still active
  - checkpoint is still not saved:
    - `/root/autodl-tmp/models-gptq-w4a16-sanity16/quantize_config.json`
  - current progress observed around:
    - `layer 22 / 31`
- Important refinement:
  - the route is no longer failing on model registration or layer mapping
  - the current errors are numerical-conditioning failures inside GPTQ
- Repeated live warnings/errors now observed:
  - `Damp recovery failed after reaching damp_percent=1.00000`
  - `Applying Hessian diagonal floor`
  - `Increasing Hessian diagonal floor`
  - `Hessian remained non positive-definite after diagonal floor attempts`
  - then:
    - `rtn failsafe`
- Current interpretation:
  - GPTQ is able to enumerate and quantize MiniCPM-SALA layers now
  - but several attention and MLP modules have badly conditioned Hessian statistics under the current calibration setup
  - the run survives by falling back to RTN on hard modules, which may let the checkpoint finish but can materially hurt quality
- Most likely causes:
  - too few calibration rows in the active sanity run:
    - `16`
  - long slices but low sample diversity:
    - large token count does not compensate for low row count in Hessian estimation
  - synthetic ultra-long rows may be amplifying correlation / instability rather than helping GPTQ statistics
  - MiniCPM-SALA's mixed attention stack:
    - `MiniCPMInfLLMv2Attention`
    - `LightningAttention`
    appears numerically fragile for generic GPTQ defaults
- Route decision:
  - do not call this a structural failure anymore
  - next quality lever is calibration quality / row count / route choice, not more `o_gate` surgery
- External tooling read:
  - `AutoGPTQ` is no longer the right mainline reference:
    - its upstream repo is archived and its README now says it is unmaintained and suggests using `GPTQModel`
  - `GPTQModel` is currently the strongest maintained open-source GPTQ path for this stack:
    - active support matrix includes GPTQ + Marlin, HF, vLLM, and SGLang
  - Open-source MiniCPM quantization material exists, but not a ready-made MiniCPM-SALA GPTQ recipe:
    - `OpenBMB/MiniCPM-CookBook` has AWQ / GPTQ / GGUF guides for MiniCPM 2.4B and 3.0 families
    - `OpenBMB/CPM.cu` ships `scripts/model_convert/gptq2marlin.py` and explicitly says MiniCPM is supported for GPTQ-to-Marlin conversion
- current implication:
  - keep `GPTQModel` as the active quantizer
  - adapt it for SALA ourselves
  - use `CPM.cu` conversion/runtime pieces as the Marlin-side companion, not as the quantizer itself

## 2026-04-09 Meaning Of `Hessian non positive-definite -> RTN failsafe`

- What it means algorithmically:
  - GPTQ is not just "round weights to 4-bit"
  - it tries to quantize each weight block while compensating quantization error using second-order curvature information
  - the core object is a Hessian-like matrix accumulated from calibration activations
  - for the usual GPTQ solve path to work cleanly, that matrix must be numerically usable for Cholesky-based inversion
- What `non positive-definite` means here:
  - the matrix that GPTQ wants to invert is too ill-conditioned / singular / noisy for Cholesky
  - in practice this usually means:
    - sample diversity is too low
    - the observed activations are too correlated
    - or the current module is numerically fragile under the calibration distribution
- What `damp recovery` and `diagonal floor` are trying to do:
  - `damp recovery`:
    - add extra diagonal damping to stabilize the matrix
  - `diagonal floor`:
    - force a minimum positive diagonal bias when plain damping still fails
  - both are "try harder to make the Hessian invertible" steps
- Why the final fallback matters:
  - when those recovery steps still fail, `gptqmodel` gives up on Hessian-based GPTQ for that module
  - it then enters:
    - `RTN` = `round to nearest`
  - `RTN` is a simple min/max-style fallback quantization path
  - it is much less informed than true GPTQ and is more outlier-sensitive
- Practical implication for our run:
  - this does **not** necessarily mean the whole run crashes
  - it means some modules are no longer getting real GPTQ treatment
  - the more modules that fall back to `RTN`, the more the final checkpoint behaves like a mixed:
    - partly-GPTQ
    - partly-RTN
    model
  - that can still finish and save, but quality can degrade sharply

## 2026-04-09 Current Read On Open-Source MiniCPM Quantization Paths

- `GPTQModel` remains the best-maintained GPTQ toolchain for our current route:
  - active version in the remote env:
    - `5.8.0`
  - current support surface explicitly includes:
    - `GPTQ`
    - `Marlin`
    - `HF`
    - `vLLM`
    - `SGLang`
- `AutoGPTQ` should not be revived:
  - upstream repo is archived
  - upstream README explicitly says it is unmaintained and suggests `GPTQModel`
- `MiniCPM-CookBook` is useful as reference material, but not as a ready SALA solution:
  - it contains quantization guides for:
    - `MiniCPM 2.4B`
    - `MiniCPM 3.0`
    - other MiniCPM family members
  - it demonstrates that OpenBMB does publish GPTQ/AWQ-style recipes for MiniCPM-family models
  - but it does not provide a drop-in `MiniCPM-SALA` GPTQ quantizer
- `CPM.cu` is relevant on the runtime/conversion side:
  - its README explicitly documents:
    - `scripts/model_convert/gptq2marlin.py`
  - and says the converter supports:
    - `MiniCPM`
    - `Llama`
    - `EAGLE`
  - this makes it a strong companion for:
    - `GPTQ checkpoint -> Marlin runtime format`
  - but it is not itself the generic quantizer we are using for SALA
- Current bottom line:
  - for `MiniCPM-SALA`, there is still no out-of-the-box open-source GPTQ quantizer
  - our real task is:
    - use `GPTQModel` as the quantizer
    - adapt model mapping / calibration strategy for SALA
    - then use `CPM.cu`-style conversion/runtime pieces where useful on the Marlin side

## 2026-04-09 First Fast/Medium Test Attempt On Saved GPTQ Checkpoint

- Saved checkpoint path:
  - `/root/autodl-tmp/models-gptq-w4a16-sanity16`
- A serialized remote test launcher was started for:
  - `fast`
  - then `medium`
- `fast` did not reach request generation.
- The blocker happened at server load time for both:
  - `--quantization gptq`
  - `--quantization gptq_marlin`
- Failure logs:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_w4a16_fast_20260409_192950/server.log`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_probe_20260409_193219/server.log`
- Common fatal error:
  - `KeyError: 'model.layers.0.self_attn.o_gate.weight'`
- Important nuance:
  - the saved checkpoint index still contains `o_gate` / `z_proj` / `q_norm` / `k_norm` / `o_norm` entries
  - so this is not simply "the file forgot to save `o_gate`"
  - instead, the likely incompatibility is:
    - `sglang`'s GPTQ/GPTQ-Marlin load path for `MiniCPM-SALA` is not reconstructing the expected `params_dict` correctly from the quantized checkpoint
    - i.e. a runtime load-schema mismatch, not just a missing tensor on disk
- Additional runtime signals from `server.log`:
  - `Detected that the model can run with gptq_marlin`
  - `gptq quantization is not fully optimized yet. The speed can be slower than non-quantized models.`
  - `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0. This might cause accuracy degradation on Blackwell.`
- Current implication:
  - the first saved GPTQ checkpoint is not yet runnable under current SGLang load paths
  - therefore:
    - no valid `fast` result exists yet
    - no `medium` run was reached
  - next real blocker is runtime compatibility:
    - patch `sglang` GPTQ load support for MiniCPM-SALA
    - or export/convert to a load path that preserves SALA-specific expected parameters end-to-end

## 2026-04-09 Runtime Compatibility Diagnosis Confirmed

- We now have a tighter diagnosis with direct code evidence.
- `MiniCPM-SALA` model code in SGLang:
  - `/root/autodl-tmp/sglang-stock/python/sglang/srt/models/minicpm.py`
- Relevant facts:
  - `o_gate` is defined as:
    - `ColumnParallelLinear`
  - `z_proj` is defined as:
    - `ColumnParallelLinear`
  - `q_norm`, `k_norm`, `o_norm` are defined as:
    - `RMSNorm`
- SGLang GPTQ selection logic:
  - `/root/autodl-tmp/sglang-stock/python/sglang/srt/layers/quantization/utils.py`
  - `get_linear_quant_method(...)` applies GPTQ methods to essentially every `LinearBase` layer unless a dynamic negative override excludes it
- Our saved checkpoint:
  - still contains float weights for:
    - `model.layers.0.self_attn.o_gate.weight`
    - `model.layers.1.self_attn.z_proj.weight`
    - and other SALA-specific non-core tensors
  - but the main GPTQ-quantized linears were exported in packed form:
    - `qweight`
    - `qzeros`
    - `scales`
    - `g_idx`
- Confirmed mismatch:
  - at runtime, SGLang treats `o_gate` / `z_proj` as quantized linear modules because they are `LinearBase`
  - but our custom GPTQ export path intentionally quantized only the common linears:
    - `q_proj`
    - `k_proj`
    - `v_proj`
    - `o_proj`
    - `mlp.{gate,up,down}_proj`
  - so runtime and checkpoint disagree about whether `o_gate` / `z_proj` should be quantized
- Why the observed `KeyError` happens:
  - `model.load_weights(...)` in `minicpm.py` eventually does:
    - `param = params_dict[name]`
  - for `name = model.layers.0.self_attn.o_gate.weight`
  - but under GPTQ/GPTQ-Marlin init, that module has already been wrapped as a quantized linear path rather than a plain `.weight` parameter
  - thus:
    - the checkpoint still offers `o_gate.weight`
    - the model-side parameter registry no longer exposes that exact name
    - runtime dies with:
      - `KeyError: 'model.layers.0.self_attn.o_gate.weight'`
- Conclusion:
  - yes, the compatibility problem is real
  - it is not a false alarm
  - it is a precise mismatch between:
    - our partial SALA-specific GPTQ export schema
    - and SGLang's default GPTQ assumption that all relevant `LinearBase` modules, including `o_gate` / `z_proj`, are quantized
- Narrow fix directions:
  - either quantize SALA-specific `LinearBase` modules too:
    - `o_gate`
    - `z_proj`
  - or patch SGLang GPTQ init/load for `MiniCPM-SALA` so those modules stay unquantized and load plain `.weight`

## 2026-04-09 Runtime Fix Applied: Keep `o_gate` / `z_proj` Unquantized

- A narrow runtime fix was applied on the remote SGLang MiniCPM-SALA model files:
  - `/root/autodl-tmp/sglang-stock/python/sglang/srt/models/minicpm.py`
  - `/root/autodl-tmp/sglang/python/sglang/srt/models/minicpm.py`
- Change:
  - when quantization is `gptq` or `gptq_marlin`
  - force:
    - `o_gate`
    - `z_proj`
    to use `quant_config=None`
  - so these SALA-specific linears remain float-weight modules and can consume the saved `.weight` tensors
- Result:
  - `gptq_marlin` readiness probe succeeded
  - the previous `KeyError: model.layers.0.self_attn.o_gate.weight` disappeared
- Active serialized run launched:
  - `fast` first, then `medium`
  - launcher log:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_fastmed_launcher.log`
  - current `fast` run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_fast_20260409_194323`
- Current state:
  - `[fast] server ready`
  - `eval_model.py` is sending `9` requests
  - `medium` will begin automatically after `fast` completes

## 2026-04-09 Quant Fast Gate Was Not Really Bounded

- The current slow `gptq_marlin` `fast` run exposed a harness bug rather than a pure model-quality issue.
- Root cause:
  - the `fast` path was still using official `eval_model.py`
  - official `eval_model.py` hardcodes `max_out_len=65536`
  - therefore the curated `fast` dataset caps were ignored at runtime
- Consequence:
  - `fast` could still behave like a tiny `medium` run with one or two decode-heavy tails
  - the observed slow tail should not be used to judge the quantized checkpoint yet
- Fixes applied locally for the next quant runs:
  - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
    - `fast-eval` now uses patched `mini_test.sh`
    - `fast-eval` auto-enables request-time stats logging
  - `/Users/ql/cursor/openbmb/scripts/run_quant_fast_medium_remote.sh`
    - quantized `fast` now also uses the bounded `mini_test.sh` path

## 2026-04-09 Stock Fast Control Refreshed

- New bounded stock control completed:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_200504_stock-base-fast-eval-rerun`
- Control result:
  - `avg_score=32.59%`
  - `pass=2`
  - `part=2`
  - `fail=5`
  - `empty=0/9`
- This is now the reference fast-gate number to compare against the next `gptq_marlin` run.

## 2026-04-09 GPTQ-Marlin Fast+Medium Relaunched

- Launched a serialized `gptq_marlin` run on the existing checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-sanity16`
- Remote script:
  - `/root/autodl-tmp/codex-drafts/run_gptq_fast_medium_remote.sh`
- Launcher log:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_fast_medium_launcher.log`
- Run dir:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_fastmed_20260409_201201`
- Sequence:
  - bounded `fast`
  - then `medium`
- Runtime observations at launch:
  - server ready under `gptq_marlin`
  - request-time stats logging enabled
  - first `fast` sample started with capped budget (`mcq req_out=64`)

## 2026-04-09 GPTQ Fast Result vs Stock Fast

- GPTQ-Marlin bounded `fast` completed with:
  - `avg_score=35.19%`
  - `pass=2`
  - `part=2`
  - `fail=5`
  - `empty=0/9`
- Stock bounded `fast` reference was:
  - `avg_score=32.59%`
- Shared fail tasks between stock and GPTQ:
  - `mcq` short
  - `niah` medium
  - `qa` long
  - `niah` long
  - `cwe` long
- Interpretation:
  - fast-gate deltas are currently small and concentrated in `fwe/cwe`
  - the shared failures are mostly not quantization-specific regressions
  - they look more like difficult long-context retrieval / extraction tasks under bounded output budgets

## 2026-04-09 GPTQ Medium Result vs Stock

- GPTQ-Marlin medium completed with:
  - `Original Accuracy: 52.80%`
  - `Normalized Accuracy: 66.0%`
  - `Total Duration: 3640.09 s`
  - `Total Output Tokens: 656170`
  - `TPS: 180.26`
- Compared with stock medium:
  - accuracy improved from `50.40%` to `52.80%`
  - normalized accuracy improved from `63.0%` to `66.0%`
  - TPS is essentially flat (`178.88` -> `180.26`)
  - duration is much worse (`2405.74s` -> `3640.09s`) because output tokens increased a lot (`430343` -> `656170`)
- Takeaway:
  - this GPTQ route currently helps quality a bit
  - but on uncapped medium-eval it over-generates, so end-to-end time gets worse even though raw token throughput is not obviously worse

## 2026-04-09 Medium Eval Dataset Reduced

- Remote medium dataset was cut from `50` to `25` samples in place for faster iteration:
  - active:
    - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl`
    - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.meta.json`
  - backups:
    - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.full50.bak`
    - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.full50.meta.bak`
- New active distribution:
  - `25` samples total
  - `5` per task
  - bucket mix:
    - `0-4K: 5`
    - `16K-32K: 8`
    - `32K-128K: 12`

## 2026-04-09 Packaging Snapshot

- Refreshed local draft package for the current GPTQ/W4A16 + Marlin route:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft`
- Reused local wheel cache instead of redownloading `flash_attn`:
  - `/Users/ql/cursor/openbmb/cache/wheels/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- Final tarball:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft.tar.gz`
  - `154M`
  - sha256:
    - `1ed9ec0e90fad4f3c6cbe27e0370d672bf1c3bda9640b6b027dca00200efe61d`

## 2026-04-09 Over-Generation Analysis

- The current `gptq_marlin` route looks like a small quality win, not a speed win:
  - fast:
    - stock `32.59%`
    - GPTQ `35.19%`
  - medium:
    - stock `50.40% / 63.0%`
    - GPTQ `52.80% / 66.0%`
- Speed read on medium:
  - stock:
    - `430343` output tokens
    - `2405.74 s`
    - `178.88 TPS`
  - GPTQ:
    - `656170` output tokens
    - `3640.09 s`
    - `180.26 TPS`
- Interpretation:
  - token throughput is basically unchanged
  - the end-to-end slowdown is dominated by much longer generations
  - this makes stopping / over-generation the most likely next target, not raw W4A16 kernel throughput
- Practical implication for the next dialogue:
  - inspect output inflation per task
  - inspect `finish_reason` and stop behavior
  - only after that decide whether runtime stop tweaks are enough or whether calibration / prompt behavior also need adjustment

## 2026-04-09 W4A16 Submission Repair

- The previous `w4a16-marlin-draft` submission failure was not a model bug.
- Root cause:
  - `prepare_model.sh` still looked for a GPTQ checkpoint that only existed on our own dev server
  - the official evaluation machine only has the original model under `/models/MiniCPM-SALA`
- Repaired package behavior:
  - `prepare_model.sh` now runs GPU-side GPTQ on the evaluation machine
  - it uses bundled calibration data and no longer depends on a prebuilt local checkpoint
- Repaired attempt order:
  1. `true160k_fp16_32`
     - calibration: `calibration_gptq_w4a16_true160k_v1.jsonl`
  2. `pg19_32k_fp16_64`
     - calibration: `calibration_gptq_w4a16_pg19_v2.jsonl`
- Repaired environment setup:
  - `prepare_env.sh` now installs, if missing:
    - `gptqmodel==5.8.0`
    - `accelerate==1.13.0`
    - `optimum==2.1.0`
    - `sentencepiece==0.2.1`
  - and still reuses the local cached `flash_attn` wheel
- This is the right direction because the platform rules only let us bring scripts and package contents, not our own prebuilt dev-box checkpoint.

## 2026-04-09 FlashAttention2 Loader Fix

- A subsequent official submission still failed inside `quantize_gptq_w4a16.py`.
- Root cause was no longer missing checkpoints.
- New blocker:
  - `transformers` detected `flash_attn` and attempted FA2 dispatch checks during model init
  - MiniCPM-SALA then failed with:
    - `ValueError: MiniCPMSALAForCausalLM does not support Flash Attention 2 yet`
- Fix applied:
  - `GPTQModel.from_pretrained(..., attn_implementation=\"eager\")`
- This is the right minimal repair because:
  - we still want `flash_attn` available later for serving
  - but GPTQ quantization-time model loading must not try to use FA2 init on this model class

## 2026-04-09 Defuser Compatibility Fix

- A later official submission still failed inside `quantize_gptq_w4a16.py`, but this time before actual GPTQ began.
- Root cause:
  - compatibility helper `maybe_disable_incompatible_defuser()` tried to import:
    - `transformers.core_model_loading`
  - when that failed, it unconditionally tried to import:
    - `defuser`
  - official machine did not have `defuser`
  - so the compatibility helper itself crashed with `ModuleNotFoundError`
- Fix applied:
  - missing `defuser` is now treated as non-fatal
  - the script logs a compatibility message and proceeds without monkey-patching `defuser`
- Packaging conclusion:
  - keep using the bundled local `flash_attn` wheel
  - keep using `uv pip` + mirror for the pure-Python quant stack
  - do **not** add `defuser` as a hard submission dependency

## 2026-04-10 Transformers 5.5.0 Quant+Fast Retest

- Requested retest target:
  - reproduce a run under `transformers 5.5.0` instead of the local success stack `4.57.1`
- Runtime decision:
  - do not touch the working shared env
  - use the dedicated `py310` env:
    - `/root/autodl-tmp/sglang/sglang_evalmatch_py310_env`
- Before launch:
  - shared env was still `transformers 4.57.1`
  - eval-match env was empty
- Env sync script was updated to install:
  - `transformers==5.5.0`
  - `gptqmodel==5.8.0`
  - `accelerate==1.13.0`
  - `optimum==2.1.0`
  - `sentencepiece==0.2.1`
  - `huggingface_hub==0.36.2`
- To avoid repeated remote timeouts on the 900MB torch wheel:
  - downloaded the `cp310 + cu128` torch wheel locally on the Mac
  - copied it to the remote host cache
- Active retest launcher:
  - session: `codex-soar`
  - window: `quant-fast-t55`
  - master log:
    - `/root/autodl-tmp/quant_fast_t55_master.log`
- Quantization plan in this retest:
  - calibration: `calibration_gptq_w4a16_pg19_v2.jsonl`
  - `64` samples
  - `float16`
  - output:
    - `/root/autodl-tmp/models-gptq-w4a16-t550-pg19-64`
- Fast plan in this retest:
  - bounded fast eval only
  - task caps:
    - `mcq:64`
    - `qa:128`
    - `niah:128`
    - `fwe:256`
    - `cwe:256`
- At the last live check the run had successfully started, but the env sync had not yet finished, so there is not yet a confirmed `transformers 5.5.0` version printout or quant/fast result.

## 2026-04-10 Transformers 5.5.0 Retest Repair Pass

- The first `t550` attempt revealed that the remote launcher still referenced an old env-sync script pinned to `transformers 4.57.1`.
- That mismatch is now repaired:
  - live launcher path:
    - `/root/autodl-tmp/remote_quant_fast_t55.sh`
  - live env-sync path:
    - `/root/autodl-tmp/remote_env_sync_evalmatch.sh`
- The next blocker was `gptqmodel==5.8.0` under build isolation.
  - Root cause:
    - pip's isolated build environment could not see installed `torch`
    - the package then raised:
      - `Unable to detect torch version via uv/pip/conda/importlib`
  - Repair:
    - install `gptqmodel==5.8.0` with `--no-build-isolation`
  - Confirmed live effect:
    - remote build now reaches the expected CUDA/Torch-specific wheel path
    - created wheel:
      - `gptqmodel-5.8.0+cu128torch2.9-cp310-cp310-linux_x86_64.whl`
- After that, the next blocker moved to editable `sglang` installation.
  - Root cause:
    - pip reopened build isolation for editable install and failed to fetch build requirements from the current mirror path
  - Repair:
    - install editable `sglang` with:
      - `--no-build-isolation --no-deps -e`
- One false shortcut was also removed:
  - `gptqmodel --no-deps` alone was not sufficient because runtime imports then failed on missing packages such as `logbar`
  - current live policy is:
    1. install pure-Python stack for `transformers 5.5.0`
    2. install `gptqmodel==5.8.0` with `--no-build-isolation`
    3. install editable `sglang` with `--no-build-isolation --no-deps -e`
- Current live status:
  - `quant-fast-t55` is still finishing env-sync dependency backfill
  - `quant-med-compare` is now waiting correctly for explicit fast completion or failure
  - no new `t550` quant/fast result yet

## 2026-04-10 Transformers 5.5.0 Quant+Fast Retest Repair Pass 2

- The next `t550` restart confirmed that the `sglang` import failure was no longer the active blocker.
- New blocker:
  - the remote draft dir was missing:
    - `calibration_gptq_w4a16_pg19_v2.jsonl`
  - immediate effect:
    - `quant-fast-t55` reached `[3/4] GPTQ quantization` and then failed on `FileNotFoundError`
- Repair:
  - resynced:
    - `calibration_gptq_w4a16_pg19_v2.jsonl`
    - `quantize_gptq_w4a16.py`
  - into:
    - `/root/autodl-tmp/codex-drafts/w4a16-marlin-draft/`
- Architectural simplification applied:
  - `transformers 5.5.0` env remains responsible for:
    - GPTQ quantization only
  - serve/eval for the retest now uses the stable shared runtime:
    - `/root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3`
  - this keeps the experiment focused on whether `t550` changes the quantized checkpoint quality, without mixing in a second `sglang` dependency chase
- Current live state after relaunch:
  - `quant-fast-t55` has re-entered `[3/4] GPTQ quantization`
  - `quant-med-compare` is waiting on the fast gate as intended

## 2026-04-10 Flash-Attn Reuse And T550 Local Repro Repair

- Confirmed the reusable binary artifact is `flash_attn`, not a standalone `fla` wheel.
- Local wheel:
  - `/Users/ql/cursor/openbmb/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- Remote wheel:
  - `/root/autodl-tmp/SOAR-Toolkit/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- Quantization script was updated to stop forcing `eager` and instead request:
  - `flash_attention_2`
  - while patching the legacy Transformers gate that still checks `_supports_flash_attn`
- Confirmed remote draft copy now uses:
  - `config._attn_implementation = "flash_attention_2"`
  - `attn_implementation="flash_attention_2"`
- The next reproduced blocker on `rtx6000-2` was not missing FA2 support, but ABI mismatch in the existing `cp310` wheel:
  - `ImportError: ... libstdc++.so.6: version CXXABI_1.3.15 not found`
- Interpretation:
  - the dev server cannot locally reproduce the `py310 + flash_attn wheel` path
  - this is an environment ABI mismatch, not a missing artifact problem
- Dev-only workaround now in progress:
  - local reproduction has been pivoted to:
    - `/root/autodl-tmp/sglang/sglang_t550_py312_env`
  - this env reuses the stable env's already working:
    - `torch`
    - `flash_attn`
    - `fla`
    - `einops`
  - submission-side `py310` reasoning remains separate
- Current live launcher:
  - session: `codex-soar`
  - window: `quant-fast-t55`
  - master log:
    - `/root/autodl-tmp/quant_fast_t55_master.log`
- Current live status:
  - the new run is active
  - it now prints:
    - `ENV=/root/autodl-tmp/sglang/sglang_t550_py312_env`
  - it is still in env-sync / package overlay before the next GPTQ load attempt

## 2026-04-10 Py312 Overlay Re-run

- The dev repro path has now been explicitly moved onto the py312 line:
  - `ENV=/root/autodl-tmp/sglang/sglang_t550_py312_env`
- The env overlay was tightened to use `--no-deps` for the `t550` package set so pip no longer tries to replace the base binary stack with a new `torch 2.11` download.
- Relaunched:
  - session: `codex-soar`
  - window: `quant-fast-t55`
  - log: `/root/autodl-tmp/quant_fast_t55_master.log`
- Latest visible state:
  - env sync has restarted in the new py312 overlay
  - no fresh giant torch download has appeared in the latest log tail after this change

## 2026-04-10 MiniCPM-SALA-Only FA2 Gate Patch

- Did not fully remove the FA2 compatibility patch.
- Reason:
  - the earlier official failure already occurred after `flash_attn` was installed
  - binary availability alone does not fix the `transformers 5.5.0` support-flag mismatch
- Narrowed the patch instead:
  - file:
    - `/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py`
    - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/quantize_gptq_w4a16.py`
  - behavior:
    - only touches MiniCPM-SALA
    - only backfills legacy `_supports_flash_attn` when `_supports_flash_attn_2` is already true
  - no longer behaves like a broad generic Transformers monkey patch
- Local `py_compile` passed after this narrowing.

## 2026-04-10 Py312 GPTQModel Build Blocker

- The current blocker is no longer `flash_attn` import/ABI.
- In the py312 repro env, `gptqmodel` is being built from source and fails during native extension build.
- Concrete failure chain from `/root/autodl-tmp/quant_fast_t55_master.log`:
  - source build enters `build_ext`
  - one internal fetch path returns `HTTP Error 404`
  - final hard failure comes from `torch.utils.cpp_extension`:
    - detected CUDA toolkit = `12.8`
    - torch compile CUDA version = `13.0`
    - runtime error: CUDA mismatch
- Interpretation:
  - this py312 repro path lacks a matching prebuilt `gptqmodel` wheel
  - local compilation is required
  - local compilation is currently impossible on `rtx6000-2` because the local CUDA toolchain and the inherited torch binary do not match
## 2026-04-10 Py310 GPTQ Environment Reset

- Evaluation probe confirmed the platform base env is `python 3.10`, not the current repo install script's `python 3.12`.
- Pivoted away from the py312 local repro line.
- New bootstrap script:
  - `/Users/ql/cursor/openbmb/scripts/install_minicpm_sala_submitmatch_uv_py310_gptq.sh`
- Goal:
  - reproduce a submission-like `py310 + uv + GPTQModel` path on `rtx6000-2`
  - use local wheels for binary pieces
  - use domestic mirror only for pure-Python packages
- Confirmed exact matching GPTQModel release wheel exists and is now cached:
  - `gptqmodel-5.8.0+cu128torch2.9-cp310-cp310-linux_x86_64.whl`
- Smoke validation already confirmed:
  - `torch` wheel install works
  - `flash_attn` wheel install works
  - `gptqmodel` wheel install works
- Remaining dev-only validation work is on the modular CUDA runtime packages required by `torch` in the fresh py310 env.

## 2026-04-10 Quant Env Import Success

- `py310 + uv + gptqmodel wheel` import smoke now passes on `rtx6000-2`.
- Validated env:
  - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env`
- Confirmed imports:
  - `torch 2.9.1+cu128`
  - `transformers 5.5.0`
  - `gptqmodel 5.8.0`
  - `accelerate 1.13.0`
  - `optimum`
  - `sentencepiece 0.2.1`
  - `huggingface_hub 1.10.1`
  - `flash_attn 2.8.3`
  - `httpx`
  - `typer`
  - `sympy`
  - `networkx`
- Important dev-only requirement:
  - must export:
    - `LD_LIBRARY_PATH=/root/miniconda3/envs/py310/lib:$LD_LIBRARY_PATH`
  - otherwise `flash_attn` fails on `CXXABI_1.3.15 not found`
- The quant-specific env split is now validated.

## 2026-04-10 Py310 GPTQ Live Run Status

- The py310 GPTQ run now advances beyond environment bootstrap and model loading.
- Confirmed blocker sequence that has already been resolved:
  - missing `fla` -> fixed by copying the existing pure-Python `fla` package from the local py312 env
  - missing `triton` -> fixed by installing `triton==3.5.1`
  - `transformers 5.5.0` FA2 gate on MiniCPM-SALA -> fixed by reintroducing a MiniCPM-SALA-only compatibility bridge in `quantize_gptq_w4a16.py`
- Current live blocker:
  - sparse attention forward now needs `infllm_v2`
  - official repo path for this is `3rdparty/infllmv2_cuda_impl`
  - `python3.10-dev` has been installed on `rtx6000-2`
  - `infllm_v2` is currently being built against:
    - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env`
- Important interpretation:
  - the end-to-end py310 path is no longer blocked by package resolution
  - it is now blocked only by the last official CUDA extension that `install_minicpm_sala.sh` would normally compile

## 2026-04-10 `infllm_v2` Build Update

- The py310 `infllm_v2` build no longer fails at the earlier `nvcc` temp-file step.
- True root cause discovered:
  - remote `/tmp` did not exist
  - this broke both `tmux` and `nvcc`
- Repair:
  - restored `/tmp` by linking it to `/root/autodl-tmp/tmp` with mode `1777`
- Current live state:
  - py310 chain restarted in tmux window `py310-chain`
  - `infllm_v2` build is actively compiling `csrc/flash_attn/flash_api.cpp`
  - no new `/tmp/tmpxft` failure has reappeared after the fix
- Next gate:
  - wait for `infllm_v2/C*.so`
  - then confirm import into both:
    - `sglang_submitmatch_uv_py310_gptq_env`
    - `sglang_submitmatch_uv_py310_eval_env`
  - after that, the same chain will continue into GPTQ quantization and bounded fast eval

## 2026-04-10 `Hessian non positive-definite / rtn failsafe` History Check

- Re-check against prior logs confirms this is not a new regression and was not previously "fixed" at the algorithm level.
- What was actually done earlier:
  - explained the warning chain and its meaning
  - moved attention from structural mapping issues toward calibration quality and row count
  - deprecated `calibration_data_merged.jsonl`
  - recommended materially more calibration rows and fewer dominating synthetic `128K-160K` rows
- Most important prior conclusion still stands:
  - the warning is mainly a calibration-statistics / Hessian-conditioning problem
  - not primarily an `o_gate` / layer-mapping problem anymore
- The current live py310 run is consistent with that historical diagnosis:
  - it is only a `16`-sample quantcheck
  - it therefore still sits in the same low-row-count regime that previously triggered RTN fallback
- Practical implication:
  - do not read current `rtn failsafe` warnings as evidence that the new py310 env is wrong
  - the environment is now substantially repaired
  - the remaining issue is the same old GPTQ numerical-stability problem under limited calibration

## 2026-04-10 GPTQ Save-Stage Tied-Weights Blocker

- Latest remote status check at `2026-04-10 11:04:38 +0800` shows the py310 GPTQ run did not stall mid-quantization:
  - it completed through `layer 31`
  - it entered `[3/3] Saving GPTQ checkpoint...`
  - then exited before producing `quantize_config.json`
  - no `gptq_py310_fastcheck_*` run started afterward
- The live blocker is now checkpoint save compatibility, not quantization progress:
  - `AttributeError: 'list' object has no attribute 'keys'`
  - traceback lands in `transformers/modeling_utils.py::_get_tied_weight_keys()`
- Narrowed root cause on the remote py310 GPTQ env:
  - remote `transformers 5.5.0` defines `_get_tied_weight_keys()` by reading `_tied_weights_keys` and calling `.keys()`
  - remote cached `MiniCPMSALAForCausalLM` still declares:
    - `_tied_weights_keys = ["lm_head.weight"]`
  - `/root/autodl-tmp/models/config.json` still says `tie_word_embeddings: false`
- Practical implication:
  - this is a MiniCPM-SALA model-definition vs `transformers 5.5.0` helper-format mismatch on the config-save path
  - it is not evidence that the quantization numerics failed after the earlier RTN-failsafe warnings
- Next minimal fix to try:
  - patch the quant script or remote draft so that just before `model.save(...)`, MiniCPM-SALA with `tie_word_embeddings=false` no longer exposes `_tied_weights_keys` as a list during `save_pretrained(..., state_dict={})`
  - if a narrow model-specific fix is insufficient, add a local compatibility monkey patch so `_get_tied_weight_keys()` accepts `list/tuple/set` in addition to `dict`

## 2026-04-10 Py310 GPTQ Completed, Fast Eval Failed In Eval Env

- User manually verified on `rtx6000-2`:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-quantcheck/quantize_config.json` exists
  - generation time is about `11:47 +0800`
- This confirms the py310 W4A16 GPTQ route passed:
  - env bootstrap
  - `infllm_v2` build/install
  - full quantization
  - checkpoint save
- The next failure moved to py310 eval runtime:
  - the auto-launched fast eval failed before requests were served
  - remote `server.log` reportedly shows:
    - `ModuleNotFoundError: No module named 'IPython'`
  - eval env involved:
    - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env`
- Current operational state after that failure:
  - no active eval / serve process on the remote host
  - required next action is operational, not algorithmic:
    - install `IPython` into the eval env
    - rerun the py310 fast eval launcher
- I attempted to execute that remotely from this Codex session, but current sandbox/network restrictions blocked SSH, so the remote operational fix still needs to be run from a network-capable shell.

## 2026-04-10 Py310 GPTQ Checkpoint Can Serve Via Dense Fallback

- The py310 W4A16 GPTQ checkpoint itself is not blocked by sparse-kernel compilation anymore.
- Direct remote probe showed:
  - `--attention-backend flashinfer --force-dense-minicpm`
  - can load the quantized MiniCPM-SALA checkpoint and bring the server to ready state
  - this bypasses `MiniCPMSparseBackend`, so it also bypasses `sparse_kernel_extension`
- Important nuance:
  - `flash_attn` is not a direct replacement for `sparse_kernel_extension` inside the sparse MiniCPM backend
  - the successful path is a dense fallback, not a sparse-kernel substitution
- Runtime fixes needed before fast eval actually ran:
  - copied `fla` into the eval env so lightning layers can use `SimpleGLAAttnBackend`
  - added eval-env `bin` to `PATH` so `flashinfer` JIT can find `ninja`
  - corrected fast harness env var names to use:
    - `API_BASE`
    - `EVAL_DATA`
- Latest live fast run using the dense fallback:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_fastcheck_20260410_130050`
- Observed live behavior:
  - server ready
  - capped fast subset correctly reduced to `9` items
  - first request completed:
    - `mcq`, `in=279`, `req_out=64`, `out=64`, `score=0.00`
  - SGLang request-time stats confirm the model is now doing real generation on the quantized checkpoint

## 2026-04-10 Py310 GPTQ Dense-Fallback Fast Result

- The py310 W4A16 GPTQ checkpoint completed a full bounded `fast` run under the dense fallback serve path.
- Final run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_fastcheck_20260410_142000`
- Final summary:
  - `avg_score=46.67%`
  - `pass=3`
  - `part=2`
  - `fail=4`
  - `empty=0/9`
- This result is materially stronger than the older `gptq_marlin fast = 35.19%` snapshot that came from the earlier route and environment.
- Interpretation:
  - the py310 checkpoint + dense fallback serve path is now good enough to use as an environment-validation submission
  - the remaining unresolved submission issue is not checkpoint quality on the repaired fast gate
  - it is whether the final package should keep using one env or explicitly separate quantization-time and eval-time stacks

## 2026-04-10 GPTQ Submission Reworked To Dual Envs

- The draft package was then upgraded from an environment-validation tarball to a formal dual-env submission candidate.
- New submission structure:
  - eval env:
    - `runtime_envs/eval_py310_env`
    - keeps `transformers 4.57.1` and the serve/runtime stack
  - quant env:
    - `runtime_envs/quant_py310_env`
    - overlays `transformers 5.5.0 + gptqmodel 5.8.0`
    - owns online GPTQ inside `prepare_model.sh`
- New packaged assets added to support this split:
  - bundled `gptqmodel` cp310 wheel
  - bundled top-level `fla/`
  - bundled `infllmv2_cuda_impl` source tree
  - bundled cp310 `infllm_v2` shared object for import-first / rebuild-second behavior
- This should be much closer to a real submission closure than the previous one-env draft because it no longer asks one environment to satisfy both:
  - SGLang serve with `transformers 4.57.1`
  - GPTQModel with `transformers >= 5.2`

## 2026-04-10 Dual-Env Bridge Fix For Official Base Torch

- A later cloud failure proved that the official evaluation image still exposes `torch`, but the new dual envs could not import it.
- The mistaken assumption was that:
  - creating overlay envs from the official base venv with `--system-site-packages`
  - would automatically expose the parent venv's installed packages
- That assumption was wrong:
  - `--system-site-packages` exposes system Python paths
  - not the parent venv's own site-packages
- Submission fix:
  - overlay envs are still used
  - but they now get an explicit `.pth` bridge into the official base env's site-packages
  - this should let both:
    - the eval env reuse official `torch`
    - the quant env reuse official `torch`
  - without rebundling a local torch wheel or pulling in mismatched CUDA runtime packages

## 2026-04-10 GPTQ Py310 Dense-Fallback Medium Eval Result

- After the py310 eval env was repaired and the quantized checkpoint was served through dense fallback, the medium eval completed:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_mediumcheck_20260410_155713`
  - outputs:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260410_155730`
- Summary:
  - `Average Score: 72.00%`
  - `Total Duration: 1256.88 s`
  - `Total Tokens: In=1494045, Out=330508`
  - `Overall TPS (Output): 262.96`
- Scope caveat:
  - this medium run uses the current `25`-row half-size subset, not the older `50`-row medium dataset
  - it is therefore not an apples-to-apples replacement for the earlier:
    - stock `50.40% / 63.0% / 2405.74 s / 178.88 TPS`
    - GPTQ `52.80% / 66.0% / 3640.09 s / 180.26 TPS`
- Breakdown on the `25` rows:
  - `fwe`: `100%`
  - `niah`: `100%`
  - `cwe`: `80%`
  - `mcq`: `40%`
  - `qa`: `40%`
- Runaway-output diagnosis remains valid:
  - `8` samples generated `323311 / 330508` output tokens (`97.82%`)
  - worst offenders were:
    - `fwe`: `65545`
    - `cwe`: `65544`, `65541`, `65538`
    - `mcq`: `27922`, `19748`, `6812`, `6661`
- Interpretation:
  - correctness on this smaller medium slice is promising
  - but stopping / over-generation is still the main performance pathology
  - especially for `cwe/fwe/mcq`

## 2026-04-10 Official Submission RoPE Default Regression Fix

- A later official submission failed again during GPTQ model loading with:
  - `ValueError: Unknown RoPE scaling type default`
- This looked familiar because the script already tried to normalize synthetic transformers-5.5 rope defaults away.
- Root cause of the regression:
  - `MiniCPMSALAConfig.rope_scaling` is effectively synthesized
  - setting only:
    - `config.rope_scaling = None`
  - changed the live attribute
  - but did **not** persist through `config.to_dict()`
- Verified behavior on the dev box:
  - before patch:
    - `getattr(config, "rope_scaling") -> {'rope_theta': 10000.0, 'rope_type': 'default'}`
    - after reset:
      - `getattr(config, "rope_scaling") -> None`
      - `config.to_dict().get("rope_scaling") -> <missing>`
  - after the stronger patch:
    - `config.__dict__["rope_scaling"] = None`
    - `config.to_dict().get("rope_scaling") -> None`
- Submission fix:
  - keep the existing synthetic-default detection
  - but persist the reset into `__dict__` as well
  - this should survive config cloning / round-trips during GPTQModel loading

## 2026-04-10 Force Quantization To Use Bundled Known-Good Model Source

- After the stronger `rope_scaling` patch, the official failure still looked the same.
- That suggested the real problem might be deeper than config normalization:
  - the official quant path could still be loading a stale cached remote-code copy
  - rather than the same `modeling_minicpm_sala.py` seen on the dev box
- Evidence:
  - the dev-box `/root/autodl-tmp/models/modeling_minicpm_sala.py` already handles:
    - `scaling_type == "default"`
  - the official traceback line numbers did not match that file's layout
- Packaging change:
  - bundle the known-good model source files from the dev box:
    - `patched_model_code/configuration_minicpm_sala.py`
    - `patched_model_code/modeling_minicpm_sala.py`
- Submission logic change:
  - `prepare_model.sh` now builds a patched local model dir
  - symlinks the original platform input files into it
  - overwrites just the Python source files with the bundled known-good copies
  - quantization then uses this patched dir as `--model-path`
  - `HF_MODULES_CACHE` is redirected under `runtime_envs/` to avoid reusing stale platform caches
- This is a stronger fix than only mutating config fields because it bypasses potentially older remote-code copies entirely.
