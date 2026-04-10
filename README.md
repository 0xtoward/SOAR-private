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

## Suggested Reading Order

1. `environment-map.md`
2. `agent-collab.md`
3. `worklog.md`
