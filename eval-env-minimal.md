# Eval Env Minimal

This note describes the smallest `py310 + uv` evaluation/serve environment we want to keep aligned with the probe from the official evaluation machine.

## Probe baseline

The probe showed the evaluation image already has:

- `python 3.10.19`
- `torch 2.9.1+cu128`
- `transformers 4.57.1`
- `sglang`
- `sentencepiece`
- `huggingface_hub`
- `tokenizers`
- `triton`

The probe did **not** have:

- `flash_attn`
- `gptqmodel`
- `modelopt`
- `accelerate`
- `optimum`

## Minimal eval-side package delta

If we reuse the base probe env via `--system-site-packages`, the strict minimal extra package is:

- `flash_attn` wheel

If we want a self-contained `uv` eval venv that is still probe-like, the practical serve/runtime additions are:

- `fastapi`
- `uvicorn`
- `uvloop`
- `aiohttp`
- `orjson`
- `msgspec`
- `numpy`
- `psutil`
- `pybase64`
- `IPython`
- `pydantic`
- `pydantic_core`
- `requests`
- `setproctitle`
- `packaging`
- `einops`
- `scipy`
- `sentencepiece`
- `huggingface_hub`
- `tokenizers`
- `safetensors`
- `tqdm`
- `python-multipart`
- `prometheus-client`
- `pyzmq`
- `pillow`

These are the smallest runtime pieces we found while tracing `sglang.launch_server` and the MiniCPM-SALA eval path. We intentionally exclude the quantization-only stack:

- `transformers 5.5.0`
- `gptqmodel`
- `modelopt`
- `accelerate`
- `optimum`

## Env split

- `eval env`: keep `transformers==4.57.1` and only add serve/runtime packages.
- `quant env`: isolate the `transformers 5.5.0 + gptqmodel` stack there, so it does not leak into serve/eval.
