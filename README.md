# OpenBMB Local Working Notes

This folder is our shared local memory for the SOAR / MiniCPM-SALA work.

## Files

- `environment-map.md`: authoritative split between local Mac, Codex global skill home, and remote `rtx6000-2`.
- `agent-collab.md`: how we record progress so both human and agent can resume work quickly.
- `worklog.md`: append-only session notes, verified findings, commands, and next steps.
- `eval-env-minimal.md`: minimal probe-aligned `py310 + uv` eval/serve package list and env split notes.

## Current Headline

- Local project root: `/Users/ql/cursor/openbmb`
- Codex skill root: `/Users/ql/.codex/skills`
- Remote host: `rtx6000-2`
- Remote workspace: `/root/autodl-tmp`
- Remote persistent tmux session when active: `codex-soar`
- Local `launchd` automation is disabled; see `environment-map.md`.

## Current Best-Known Recipe

Until newer evidence clearly beats it, treat the current best-known MiniCPM-SALA quantized route as:

- Weight recipe:
  `W4A16 GPTQ-Marlin` with MLP-heavy quantization plus a small attention-QKV allowlist.
- MLP policy:
  quantize all `gate_proj` and `up_proj`, and quantize all `down_proj` except layers `15/16/17`, which stay FP16.
- Attention policy:
  quantize only `self_attn.q_proj/k_proj/v_proj` for layers `10/12/13/14/19/20/21/23`.
- Attention modules kept dense/full-precision:
  all `o_proj` remain unquantized; do not assume a fully quantized attention stack.
- Runtime baseline:
  `--disable-radix-cache --attention-backend flashinfer --force-dense-minicpm --chunked-prefill-size 8192 --skip-server-warmup --disable-cuda-graph --quantization gptq_marlin --dtype float16`

This is the default reference point for new runtime experiments and submission siblings unless a newer note in `worklog.md` explicitly supersedes it.

## Suggested Reading Order

1. `environment-map.md`
2. `agent-collab.md`
3. `worklog.md`
