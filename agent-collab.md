# Agent Collaboration Notes

## Environment split

- Local Mac control plane:
  - `/Users/ql/cursor/openbmb`
- Codex global skills:
  - `/Users/ql/.codex/skills`
- Remote GPU execution plane:
  - `rtx6000-2:/root/autodl-tmp`

Read `environment-map.md` first if there is any ambiguity about where a path or process lives.

## Why keep local notes

Yes, this is good agent practice.

The useful pattern is:

1. Keep stable facts separate from daily work.
2. Keep an append-only worklog with evidence, not only conclusions.
3. Record the exact remote host, path, command, and log file for anything we claim is verified.

This makes handoff easier for:

- you reading in the IDE
- me resuming later
- future agents continuing without re-discovering the same state

## What goes where

- Stable facts:
  environment paths, model versions, known-good baselines, evaluation rules
- Worklog:
  what we tried today, what happened, log paths, next actions
- Decisions:
  short statements like "runnable but inaccurate" with the evidence right below them

## Entry format

Use short entries with:

- `Date`
- `Goal`
- `What was verified`
- `Evidence`
- `Next actions`

## Rules for useful entries

- Prefer exact paths over vague descriptions.
- Prefer measured results over impressions.
- Record one-line conclusions, then the supporting evidence.
- If a run uses a remote machine, include the host and log path.
- If a result is only a smoke test, say so explicitly.

## Current convention

- This local folder is the human/agent handoff layer.
- Remote experiment artifacts still live on `rtx6000-2` under `/root/autodl-tmp/SOAR-Toolkit`.
- Local automation logs, when relevant, live under `/Users/ql/cursor/openbmb/automation/`.
- Codex reusable skills do not live in this repo; they live under `/Users/ql/.codex/skills/`.
