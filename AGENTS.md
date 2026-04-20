# OpenBMB Project Rules

These rules are project-specific and should be followed for SOAR / OpenBMB work in this workspace.

## Remote Job Policy

- Do not launch long-running remote work via plain foreground `ssh`.
- All remote jobs that launch servers, run evals, quantize models, benchmark kernels, package submissions, or write substantial artifacts must run inside `tmux` or under `nohup`.
- Prefer `tmux` for:
  - `launch_server`
  - `mini_test.sh`
  - `eval_model.py`
  - `fast` / `medium` eval
  - quantization
  - long benchmark or sweep scripts
- Foreground `ssh` is only for short read-only inspection:
  - checking logs
  - checking `nvidia-smi`
  - checking file presence
  - small status commands
- After starting any remote job, always record and report:
  - remote host / alias
  - remote working directory
  - tmux session and window, or background PID
  - log file path
  - run directory
  - exact command

## Server / Target Stability

- Do not switch GPT server, serving endpoint, remote clone host, SSH alias, model-serving host/port, or eval target unless the user explicitly asks.
- Do not silently change between different `rtx6000-*` targets.
- Do not silently change API endpoints, base URLs, model ids, or server ports.
- Do not silently change the active package, probe directory, or remote root.
- If the active target is unhealthy, collect evidence first, then ask or clearly justify the fallback before switching.

## SOAR-Specific Safety

- Freeze the current winning package unless the user explicitly asks to modify it.
- Risky runtime or KV-cache work must stay in an isolated probe directory.
- Runtime-only probes should usually touch the eval env only.
- Do not mix environment repair, server-target switching, and model-quality conclusions into one silent change.
