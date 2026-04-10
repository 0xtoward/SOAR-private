# SOAR Notes Index

## Purpose

This folder stores project knowledge for SOAR 2026 in a stable structure.
Use these files instead of the old monolithic `INSIGHTS.md`.

## Files

- `project-facts.md`: Stable platform facts, limits, default args, score formula, and official data distribution.
- `model-notes.md`: SALA architecture notes, backend trade-offs, and optimization directions.
- `testing-playbook.md`: Local testing SOP for service startup, fast eval, fast bench, packaging, and result collection.
- `experiment-log.md`: Recent experiments and submission results. Keep this short and current.
- `roadmap.md`: Current priorities and next steps.
- `champion/`: Raw week-by-week champion notes and summaries kept for reference.

## Suggested Reading Order

1. Read `project-facts.md` for hard constraints.
2. Read `testing-playbook.md` before running any eval or bench.
3. Read `model-notes.md` before editing `minicpm.py` or changing backends.
4. Read `experiment-log.md` to understand the latest validated state.
