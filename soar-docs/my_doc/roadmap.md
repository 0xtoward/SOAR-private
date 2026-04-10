# Roadmap

## P0

- Finish the new realistic fast bench and compare it with the official v2 platform result.
- Stabilize the fast-test loop so every code change gets one smoke eval plus one realistic bench.

## P1

- Try GPTQ with real calibration data instead of RTN.
- Investigate NVFP4 feasibility on Blackwell.

## P2

- Expand code-level optimizations beyond the current fused norm and rotary cleanup.
- Add a lightweight logprobs-based validation tool for faster iteration between full eval runs.

## P3

- Compare `flashinfer` and `minicpm_flashinfer` with the new bench set.
- Revisit whether any backend-specific tuning helps S1 more than S8 and Smax.
