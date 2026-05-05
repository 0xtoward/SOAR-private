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

## Current Review Branch

This branch is a lightweight, reviewable companion to the SGLang source branch.
It intentionally does **not** include model weights, generated tarballs, wheels,
calibration JSONL payloads, or a vendored SGLang tree.

The current overlay is:

- `submission_overlays/allqkv189-bf16full115-runtimebf16-v1/`

It documents the non-SGLang submission-side changes for the SOAR MiniCPM-SALA
experiment:

- expand W4A16 GPTQ-Marlin coverage from the earlier 117-module route to
  `all-QKV189`;
- use `bfloat16` for the quantization/calibration forward;
- serve the resulting checkpoint with runtime `--dtype bfloat16`;
- keep the high-risk modules conservative: all `self_attn.o_proj` stay dense,
  and `layers.15/16/17.mlp.down_proj` stay dense.

SGLang runtime/source changes are kept in a separate branch for a clean GitHub
compare against `OpenBMB/sglang:minicpm_sala`:

- `0xtoward/sglang:codex/minicpm-sala-clean-diff-20260505`

## Suggested Reading Order

1. `environment-map.md`
2. `agent-collab.md`
3. `worklog.md`
4. `submission_overlays/allqkv189-bf16full115-runtimebf16-v1/README.md`
