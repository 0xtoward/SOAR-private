# SimpleGLA Notes

This folder collects the current MiniCPM-SALA `SimpleGLA` profiling and optimization notes.

## Files

- `nsight-rtx6000-1-env-2026-04-20.md`
  - remote `rtx6000-1` Nsight environment notes
  - what works (`nsys`) and what is still blocked (`ncu` counters)
- `simplegla-nsight-analysis-2026-04-20.md`
  - current system-level visual analysis
  - how to read the Nsight traces
  - where the real hotspots are in prefill vs decode
- `indexed-fused-decode-results-2026-04-20.md`
  - decode-side `SimpleGLA` optimization summary
  - negative and positive experiments
  - current recommendation for end-to-end use

## Current Bottom Line

- `SimpleGLA` is not the main prefill bottleneck.
- The meaningful hotspot is the decode-side state path:
  - state gather
  - recurrent update
  - state writeback
- Wrapper-level locality tricks did not help end-to-end.
- A decode-only indexed fused recurrent/state-io fast path did help end-to-end:
  - `float32 state`: about `+4.86%` total tok/s
  - `bfloat16 state`: about `+7.36%` total tok/s
- `nsys` remains usable on `rtx6000-1`.
- `ncu` is present, but GPU performance counters are still blocked by permissions on the current host/container.
