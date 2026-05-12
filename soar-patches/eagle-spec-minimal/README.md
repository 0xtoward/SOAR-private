# MiniCPM-SALA EAGLE3 Minimal Runtime Patch

This branch keeps the EAGLE3 work small and runnable:

- `submission-drafts/w4a16-marlin-draft/sglang/...` contains the minimal SGLang runtime changes required for MiniCPM-SALA EAGLE3 serving.
- `stream_eval_model.py` is a SOAR-compatible streaming eval wrapper that writes per-request predictions and partial accuracy while requests finish.
- `launch_eagle3_stream_eval_remote.sh` is a local orchestrator for the SeetaCloud/AutoDL-style remote box. It launches serving/eval in `tmux` and records all logs under the SOAR `test_results` directory.

Large artifacts are intentionally not committed:

- target model checkpoints
- quantized model directories
- EAGLE3 draft/head checkpoints
- generated eval outputs

The default launch values match the current winning99+ EAGLE smoke path, but all important paths and speculative parameters are overridable by environment variables.

## Minimal Remote Run

```bash
REMOTE_HOST=root@connect.bjb1.seetacloud.com \
REMOTE_PORT=14968 \
PKG=/root/autodl-tmp/codex-drafts/submission-w4a16-marlin-attnqkv-allowlist-gate0fixorder-cgbs1-fp8-kvscale-mambabf16-v1 \
MODEL_PATH=/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder \
DRAFT_PATH=/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/outputs/main2000_full_vocab_lr5e5_ttt5/epoch_0_step_1600 \
KV_SCALE_JSON=/root/autodl-tmp/SOAR-Toolkit/test_results/kvscale_input_full_20260421_160936/kv_scales.json \
SPEC_STEPS=3 \
SPEC_TOPK=1 \
SPEC_DRAFT_TOKENS=4 \
MAX_OUT_LEN=32768 \
CONCURRENCY=8 \
bash soar-patches/eagle-spec-minimal/launch_eagle3_stream_eval_remote.sh
```

Useful smoke variants:

```bash
# Tree verify smoke. Queue it after a current run by setting WAIT_FOR_SESSION.
WAIT_FOR_SESSION=codex_winning99_eagle3_eval_32k \
SESSION=codex_winning99_eagle3_topk2_stream_smoke \
RUN_ID=winning99_eagle3_stream_topk2_smoke \
SPEC_TOPK=2 \
SPEC_DRAFT_TOKENS=5 \
NUM_SAMPLES=16 \
MAX_OUT_LEN=4096 \
CONCURRENCY=4 \
bash soar-patches/eagle-spec-minimal/launch_eagle3_stream_eval_remote.sh
```

## Runtime Notes

- The target model uses `MiniCPMSALAForCausalLM`; the EAGLE3 worker expects `get_embed_and_head()` and `set_eagle3_layers_to_capture()`.
- MiniCPM-SALA has SimpleGLA recurrent temporal state in addition to standard KV cache.
- During speculative `TARGET_VERIFY`, SimpleGLA states are computed into speculative intermediate caches. The accepted step is committed after verification.
- Tree verify uses `retrieve_next_token` and `retrieve_next_sibling` to derive parent steps so sibling branches do not inherit one another's recurrent state.
- `SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE=1` is the default in the launcher. It avoids prematurely committing the final verify node before EAGLE chooses the accepted step.
