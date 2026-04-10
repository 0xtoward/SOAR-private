# Project Facts

## Platform And Limits

- GPU target: NVIDIA RTX PRO 6000 Blackwell, about 96 GB VRAM.
- Resource limit per evaluation: 20 CPU, 128 GiB RAM.
- Max execution time per submission: 5 hours.
- Submission package size must stay under 2 GB.
- `prepare_env.sh` is sourced by the platform. Do not use `set -euo pipefail` there.
- `prepare_model.sh` is executed by `bash` and must support `--input` and `--output`.
- Package manager is `uv`, so use `uv pip install`, not plain `pip install`.

## Safe Default Server Args

```bash
SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse"
```

- Parameter names must use hyphens, for example `--dense-as-sparse`.
- Prefix cache is disabled in official evaluation, so local tests should keep `--disable-radix-cache`.

## Environment Paths

- Toolkit repo: `/root/autodl-tmp/SOAR-Toolkit`
- SGLang source: `/root/autodl-tmp/sglang`
- SGLang venv: `/root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/activate`
- Model path: `/root/autodl-tmp/models/`
- Submission dir: `/root/autodl-tmp/SOAR-Toolkit/submissions/v3-fusion-flashinfer`

## Attention Backend Facts

- `minicpm_flashinfer`: baseline backend, keeps sparse TopK behavior.
- `flashinfer`: usually faster on standard attention, but may lose some sparse behavior and needs accuracy validation.
- Linear attention layers still go through `SimpleGLAAttnBackend`, so backend switching mainly affects standard attention layers.

## Accuracy Coefficient

- `acc <= 97%` -> `C = 0`
- `97% < acc <= 98%` -> `C = 0.92`
- `98% < acc <= 99%` -> `C = 0.96`
- `99% < acc <= 100%` -> `C = 1.0`

Final score:

```text
performance_score = S1 * 0.4 + S8 * 0.3 + Smax * 0.3
final_score = performance_score * C
```

## Official Speed Distribution

### Input Length

- `0-4K`: about 25%
- `4K-16K`: about 10%
- `16K-32K`: about 15%
- `32K-128K`: about 35%
- `128K-160K`: about 15%

### Output Length

- `0-512`: about 35%
- `512-2K`: about 25%
- `2K-4K`: about 10%
- `4K-16K`: about 15%
- `16K-32K`: about 15%

## Known Platform Caveats

- FP8 KV cache is not usable with the platform flashinfer FA2 backend.
- Enabling `--kv-cache-dtype fp8_e5m2` caused startup failure in local tests.
