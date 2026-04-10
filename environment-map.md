# Environment Map

Date: 2026-04-09

## Local Mac

- Role:
  - control plane
  - local notes, scripts, automation logs, handoff docs, submission drafts
- Root:
  - `/Users/ql/cursor/openbmb`
- Key local paths:
  - `/Users/ql/cursor/openbmb/worklog.md`
  - `/Users/ql/cursor/openbmb/quantization-log.md`
  - `/Users/ql/cursor/openbmb/sgl-optimization-log.md`
  - `/Users/ql/cursor/openbmb/marlin-route-log.md`
  - `/Users/ql/cursor/openbmb/clone-server-handoff.md`
  - `/Users/ql/cursor/openbmb/scripts/`
  - `/Users/ql/cursor/openbmb/submission-drafts/`
  - `/Users/ql/cursor/openbmb/automation/`

## Codex Skill Home

- Role:
  - reusable skills outside the repo
- Root:
  - `/Users/ql/.codex/skills`
- Relevant skill:
  - `/Users/ql/.codex/skills/soar-subagent-orchestrator/SKILL.md`

## Remote RTX6000 Server

- Role:
  - actual GPU execution
  - SGLang serving
  - quantization
  - eval / bench artifacts
- Host:
  - `rtx6000-2`
- Root:
  - `/root/autodl-tmp`
- Key remote paths:
  - `/root/autodl-tmp/SOAR-Toolkit`
  - `/root/autodl-tmp/models`
  - `/root/autodl-tmp/sglang`
  - `/root/autodl-tmp/sglang-stock`
  - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env`
  - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env`
  - `/root/autodl-tmp/SOAR-Toolkit/test_results`
  - `/root/autodl-tmp/SOAR-Toolkit/outputs`
  - `source /root/autodl-tmp/use_soar_uv.sh`

## Remote Env Rule

- Remote GPU work now has two separate Python environments on purpose.
- Quantization env:
  - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env`
  - use only for:
    - GPTQ / W4A16 model preparation
    - `transformers 5.5.0`
    - `gptqmodel`
- Eval env:
  - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env`
  - use only for:
    - SGLang serving
    - `fast` / `medium` eval
    - `transformers 4.57.1`
- Do not merge these two roles into one env again.

## Local Automation Status

- Former LaunchAgent:
  - `/Users/ql/Library/LaunchAgents/com.ql.codex.soar.resume30m.plist`
- Actual role:
  - local Mac only
  - periodically ran `codex exec resume ...` against this conversation
  - wrote logs under `/Users/ql/cursor/openbmb/automation/logs/`
  - it did not itself run on the remote GPU host
- Current status:
  - stopped and disabled on 2026-04-09
- Disabled file:
  - `/Users/ql/Library/LaunchAgents/com.ql.codex.soar.resume30m.plist.disabled`
