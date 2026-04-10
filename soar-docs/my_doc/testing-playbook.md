# Testing Playbook

## Purpose

Use this file as the default SOP for local validation before any SOAR submission.

## Test Artifacts

- Fast test script: `/root/autodl-tmp/SOAR-Toolkit/fast_test.sh`
- Test logs: `/root/autodl-tmp/SOAR-Toolkit/test_results/`
- Fast bench smoke dataset: `/root/autodl-tmp/SOAR-Toolkit/bench_speed_v2_smoke.jsonl`
- Fast bench proxy dataset: `/root/autodl-tmp/SOAR-Toolkit/bench_speed_v2_mini.jsonl`
- Fast eval dataset: `/tmp/fast_eval_mini.jsonl`

## Deploy Current Submission Code

```bash
cp /root/autodl-tmp/SOAR-Toolkit/submissions/v3-fusion-flashinfer/sglang/python/sglang/srt/models/minicpm.py \
   /root/autodl-tmp/sglang/python/sglang/srt/models/minicpm.py
```

Restore baseline:

```bash
cp /root/autodl-tmp/sglang/python/sglang/srt/models/minicpm.py.orig \
   /root/autodl-tmp/sglang/python/sglang/srt/models/minicpm.py
```

## Start Service

```bash
source /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/activate
python3 -m sglang.launch_server \
  --model-path /root/autodl-tmp/models/ \
  --host 0.0.0.0 --port 30000 \
  --disable-radix-cache \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

Wait until the log shows `The server is fired up and ready to roll!`.

## Run Fast Eval

```bash
bash /root/autodl-tmp/SOAR-Toolkit/fast_test.sh --eval-only
```

Current fast eval strategy:

- 10 samples total
- 2 shortest samples per task
- meant as a smoke test, not an official predictor

## Run Fast Bench

```bash
bash /root/autodl-tmp/SOAR-Toolkit/fast_test.sh --bench-only
```

Default fast bench dataset:

- `bench_speed_v2_smoke.jsonl`
- intended for daily regression
- target runtime: under about 5 minutes locally
- keeps long-input and long-output pressure, but skips the most extreme tail

Proxy bench dataset:

- `bench_speed_v2_mini.jsonl`
- closer to the official distribution
- use before submission or when comparing serious candidates

## Important Caution

- Eval may leave the server in a bad memory state after long generations.
- If eval is long or unstable, restart SGLang before bench.

## Reference Local Baselines

Old small mini-bench baseline, BF16 default params:

| Tier | Duration | Output TPS |
|---|---|---|
| S1 | 175.80 s | 52.1 tok/s |
| S8 | 46.71 s | 196 tok/s |
| Smax | 39.69 s | 231 tok/s |

These numbers are useful only as a local comparison anchor. They do not match the official hidden speed set.

## Package Submission

```bash
cd /root/autodl-tmp/SOAR-Toolkit/submissions
tar czf v3-fusion-flashinfer.tar.gz v3-fusion-flashinfer/
ls -lh v3-fusion-flashinfer.tar.gz
```

## Result Recording Template

When a run completes, append a short entry to `experiment-log.md` with:

- date
- code version
- dataset
- accuracy result
- bench result
- whether the result is only a smoke test or suitable for comparison
