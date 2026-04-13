# Current Results Snapshot

This draft reflects the current locally validated state of the `GPTQ W4A16 + Marlin` route.

At the moment this draft is structured as a formal dual-env submission candidate for the py310 evaluation image.

## Submission Materialization

- This package no longer assumes a GPTQ checkpoint already exists on our own dev server.
- `prepare_model.sh` now quantizes from the platform-provided original model:
  - input:
    - `/models/MiniCPM-SALA`
  - output:
    - platform-provided prepared model directory
- `prepare_env.sh` explicitly pins the core quantization stack to the locally validated versions:
  - eval env:
    - reuse official image `torch==2.9.1+cu128`
    - `transformers==4.57.1`
    - `flash_attn`
    - editable `sglang/python`
  - quant env:
    - reuse official image `torch==2.9.1+cu128`
    - bundled `gptqmodel==5.8.0+cu128torch2.9` wheel
    - `transformers==5.5.0`
    - `accelerate==1.13.0`
    - `optimum==2.1.0`
    - `sentencepiece==0.2.1`
- Current quantization settings:
  - `bits=4`
  - `group_size=128`
  - `quant_method=gptq`
  - `checkpoint_format=gptq`
  - `dtype=float16`
  - dynamic exclusions:
    - `self_attn.o_gate`
    - `self_attn.z_proj`
- Attempt order:
  1. `stopaligned64k_v1`
  2. `true160k_fp16_32`
  3. `pg19_32k_fp16_64`

## Runtime Compatibility

The current SGLang runtime requires one MiniCPM-SALA-specific compatibility patch:

- keep `o_gate` and `z_proj` unquantized when `quantization in {gptq, gptq_marlin}`

Without that patch, the current checkpoint fails to load with:

- `KeyError: model.layers.0.self_attn.o_gate.weight`

The currently validated py310 serve path is also a dense fallback:

- `--attention-backend flashinfer`
- `--force-dense-minicpm`

This is not the original sparse `minicpm_flashinfer` route. It is the fallback that is currently verified to boot and serve GPTQ on the py310 evaluation-style environment.

The key submission design decision is now explicit:

- eval env and quant env are separated on purpose
- `prepare_env.sh` should stay close to the official py310 probe image
- `prepare_model.sh` owns the `transformers 5.5.0 + gptqmodel` overlay

## Bundled Calibration

- Primary calibration candidate:
  - `calibration_gptq_w4a16_stopaligned64k_v1.jsonl`
  - `64` rows
  - `36` SOAR public rows + `16` SOAR chat-close rows + `12` PG19 local rows
  - intended effect:
    - preserve assistant-closing / `<|im_end|>` states without changing eval logic
- Primary calibration:
  - `calibration_gptq_w4a16_true160k_v1.jsonl`
  - `64` rows
  - `48` SOAR public rows + `16` PG19 rows
  - `p50=63410`
  - `p90=160000`
  - `max=160000`
- Fallback calibration:
  - `calibration_gptq_w4a16_pg19_v2.jsonl`
  - `64` rows
  - `48` SOAR public rows + `16` PG19 rows
  - `p50=32768`
  - `p90=32768`
  - `max=32768`

## Current Fast Result

Bounded `fast` on stock base:

- `avg_score=32.59%`
- `pass=2`
- `part=2`
- `fail=5`
- `empty=0/9`

Bounded `fast` on the latest py310 dense fallback `gptq_marlin` eval env:

- `avg_score=46.67%`
- `pass=3`
- `part=2`
- `fail=4`
- `empty=0/9`

Bounded `fast` on `stopaligned64k_v1` under the same py310 dense fallback eval env:

- `avg_score=58.89%`
- previous py310 GPTQ baseline:
  - `46.67%`

Interpretation:

- `gptq_marlin` is materially ahead of stock on the repaired bounded fast gate
- most failures are still shared between stock and GPTQ
- the new py310 dense fallback route is usable enough to test evaluation-environment issues without depending on the sparse MiniCPM runtime
- the stop-aligned calibration candidate improved the bounded fast gate materially over the earlier py310 GPTQ baseline

## Current Medium Result

Stock base `medium`:

- `Original Accuracy: 50.40%`
- `Normalized Accuracy: 63.0%`
- `Total Duration: 2405.74 s`
- `Total Output Tokens: 430343`
- `TPS: 178.88`

`gptq_marlin` `medium`:

- `Original Accuracy: 52.80%`
- `Normalized Accuracy: 66.0%`
- `Total Duration: 3640.09 s`
- `Total Output Tokens: 656170`
- `TPS: 180.26`

`stopaligned64k_v1` on the current `25`-row half-medium file:

- `Average Score: 84.40%`
- `Total Duration: 1206.01 s`
- `Total Output Tokens: 207140`
- `TPS: 171.76`
- previous py310 GPTQ baseline on the same file:
  - `Average Score: 72.00%`
  - `Total Duration: 1256.88 s`
  - `Total Output Tokens: 330508`
  - `TPS: 262.96`

Interpretation:

- this GPTQ route is currently a modest quality win over stock on the validated medium set
- raw throughput is roughly flat
- wall-clock is worse because the quantized model over-generates on uncapped `medium`
- the stop-aligned calibration candidate improved score and reduced total output tokens materially on the current `25`-row medium slice
- however, it is still not a fully closed final route because quantization remained numerically risky (`221` RTN failsafe modules) and `cwe` is still the clearest remaining over-generation weakness

## Current Read

This route is now worth keeping and packaging as a formal dual-env submission candidate.

The main remaining weakness is not obvious raw throughput collapse. It is:

- stopping / over-generation

So the next likely gains are in:

- better stop behavior
- output-length control
- answer-format brevity

not in changing the route away from `GPTQ + Marlin`.
