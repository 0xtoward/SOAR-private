# Experiment Log

This file keeps only the latest useful records.
Old exploratory notes were intentionally cleared during the refactor.

## 2026-03-27 - v8 Phase: W4A16 GPTQ 预量化模型 fast_test 完成

**Status**: 第三方预量化模型精度不足（30-34%），放弃此候选

### 测试结果

| 配置 | avg_score | tps | mem | 备注 |
|---|---|---|---|---|
| gptq_marlin | 30.00% | 4.63 | 6.52GB | Marlin kernel |
| gptq (非 marlin) | 34.00% | 4.65 | ~6.5GB | 标准 GPTQ kernel |
| v3-fusion-flashinfer | 97.00% | ~4.72 | 18GB | 本地 no-trunc |

### 关键发现

1. **SGLang + gptq_marlin 加载链完全可行**
   - 权重加载成功：18GB → 6.52GB
   - gptq_marlin kernel 正常运行
   - 服务稳定运行

2. **第三方预量化模型精度太差**
   - 模型: `ericzhang0328/MiniCPM-SALA-GPTQ-w4a16-ctx4096`
   - mcq/niah/qa 全部 0 分
   - cwe 部分正确（0.50-0.70）
   - fwe 完美（1.00）
   - 根本原因：校准数据不适合 SOAR 评测集

3. **gptqmodel 自己量化仍然被阻塞**
   - gptqmodel 5.8.0 需要 `transformers.core_model_loading`（transformers 4.58+）
   - 当前 transformers 4.57.1 缺少此模块
   - 升级 transformers 可能破坏 SGLang 兼容性

### 结论

- **GPTQ Marlin 路径技术可行性已验证**
- **但需要自己用高质量校准数据量化**（冠军强调的关键点）
- 自己量化被 gptqmodel + transformers 兼容性阻塞
- **当前最佳提交**: v3-fusion-flashinfer (final_score=37.36) 保持未修改

### 下一步

- 尝试升级 transformers 到 4.58+ 看是否能解除 gptqmodel 阻塞
- 或评估 NVFP4/modelopt 路径（需要 sgl_kernel 编译）
- 或保持 v3-fusion-flashinfer 不变

---

## 2026-03-27 - v8 Phase: W4A16 GPTQ 预量化模型测试

**Status**: 模型下载完成，环境准备就绪，flashinfer 安装完成，等待测试

### 关键发现

1. **发现 HuggingFace 预量化模型**
   - 仓库: `ericzhang0328/MiniCPM-SALA-GPTQ-w4a16-ctx4096`
   - 配置: bits=4, group_size=128, sym=True, desc_act=False
   - 敏感层: o_gate 和 z_proj 保持 FP16（混合精度策略）
   - 总大小: ~6.2GB（原始 18GB → 6.2GB，压缩率约 65%）

2. **模型配置完全符合 Week 4 冠军路线**
   - W4A16 + GPTQ + Marlin
   - 敏感层保持高精度
   - desc_act=False → 支持 Marlin kernel 加速

3. **工具链进展**
   - sgl-kernel 0.3.18 安装成功（兼容 torch 2.8.0）
   - gptq_marlin_gemm kernel 可用
   - flashinfer_python 0.5.3 安装成功

### 阻塞链

```
原始 HF 模型量化 → flash_attn 强制要求 → torch 2.9.1/python 3.12 ABI 不兼容
                    ↓
            使用预量化 GPTQ 模型（绕过量化步骤）
                    ↓
          SGLang 启动需要 flashinfer → pip install 超时
                    ↓
               安装 flashinfer_python 成功
                    ↓
              准备测试 SGLang 加载
```

### 下一步

1. 验证 SGLang 能否加载预量化 GPTQ 模型
2. 运行 fast_test.sh smoke 测试
3. 如成功，创建 v8-gptq-w4a16 提交目录

### 当前最佳提交

**v3-fusion-flashinfer** (官方 final_score=37.36) 保持未修改

---

## 2026-03-27 - v8 Phase: 所有量化路径阻塞

**Status**: 全部阻塞，回到 v3-fusion-flashinfer

### 阻塞链

```
NVFP4 量化 → sgl_kernel 编译超时
modelopt 独立量化 → HF AutoModel → flash_attn 强制要求
flash_attn → torch 2.9.1/python 3.12 ABI 不兼容
```

### 尝试过的方案

1. **SGLang 官方 modelopt_quantize_and_export.py**
   - 依赖 sgl_kernel==0.3.20，编译安装超时
   - 需要约 30+ 个 pip 依赖

2. **modelopt 独立量化（NVFP4/FP8）**
   - modelopt 0.42.0 已安装，quantization API 可用
   - 但 MiniCPM-SALA HF 建模代码强制要求 flash_attention_2
   - flash_attn 2.8.3 与 torch 2.9.1/python 3.12 ABI 不兼容
   - 错误：`undefined symbol: _ZNK3c106SymInt22maybe_as_int_slow_pathEv`

3. **Python 3.10 conda 环境**
   - 创建成功，但 torch 2.9.1 安装超时（下载慢）

### 结论

- **当前最佳提交**: v3-fusion-flashinfer (官方 final_score=37.36)
- 提交包大小: 172M，远低于 2GB 限制
- 所有量化路径需要修复 flash_attn 兼容性
- 详细阻塞分析: `submissions/experiments/BLOCKER.md`

---

## 2026-03-26 - AWQ Marlin + flashinfer Backend 实验总结

**Status**: AWQ 路径无速度提升，NVFP4 工具链不可用

### 测试结果

| Model         | Backend           | avg_score | tps   | size  |
|---------------|-------------------|-----------|-------|-------|
| Baseline      | minicpm_flashinfer| 28%       | 4.72  | 18G   |
| v6 AWQ        | minicpm_flashinfer| 32%       | 4.72  | 15G   |
| v6 AWQ        | flashinfer        | 12%       | 8.14  | 15G   |
| v7 AWQ        | minicpm_flashinfer| 30%       | 4.64  | 7.6G  |

### 关键发现

1. **AWQ 量化对 MiniCPM-SALA 速度提升无效**
   - minicpm_flashinfer 后端下 AWQ 量化无提速
   - v6 (中层 gate/up 量化) 和 v7 (激进量化) 速度与 baseline 相同
   - 原因：模型大部分计算仍在 FP16，MLP 量化收益有限

2. **flashinfer 后端速度提升但精度下降**
   - v6 AWQ + flashinfer: tps=8.14 (快72%)
   - 但 avg_score 从 32% 降到 12%
   - 原因：MiniCPM-SALA 的 sparse TopK attention 行为丢失

3. **NVFP4 路径阻塞**
   - nvidia-modelopt 未安装
   - petit_kernel 不可用
   - SGLang quantize_and_serve 功能被禁用
   - flashinfer nvfp4_block_scale_interleave 可用，但需要完整工具链

4. **SGLang 支持的量化方法**
   - `awq_marlin`: 可用，但对 MiniCPM-SALA 无提速
   - `modelopt_fp4`: 需要安装 nvidia-modelopt
   - `petit_nvfp4`: 需要安装 petit_kernel
   - `w4afp8`: 需要特定格式权重
   - `mxfp4`: 需要 compressed-tensors 格式

### 结论

- **当前最佳提交**: v3-fusion-flashinfer (官方 final_score=37.36)
- v6 AWQ 可作为备选方案但无速度优势
- 需要 NVFP4 工具链才能实现真正的 W4 加速
- v3-fusion-flashinfer 保持未修改 ✓

### Next step

- 如需继续量化研究，需安装 nvidia-modelopt 或 petit_kernel
- 或尝试 flashinfer + 更大截断来平衡速度与精度
- 当前建议保持 v3-fusion-flashinfer 作为稳定提交

---

## 2026-03-25 - New Prototype: Selective RTN GPTQ Submission

**Status**: Prototype created, quantization/load/start verified, quality currently failed

### What changed

- Added `submissions/v4-selective-rtn-gptq/`
- This starts from official `demo-quant`, but changes the strategy from full RTN W4A16 to selective mixed precision:
  - keep `o_gate` and `z_proj` in FP16
  - keep first `2` and last `2` layers at `W8A16`
  - quantize the remaining linear layers to `W4A16`

### Why

- Champion Week 2 notes explicitly say the winning direction was:
  - sensitive layers at higher precision
  - the rest at W4
- Our old `v1-gptq-failed` path used full RTN W4A16 and collapsed to `acc=0`
- This new prototype aims to keep the **competition-valid `prepare_model.sh` route** while avoiding the most aggressive all-layer W4 setting

### Validation

- `prepare_env.sh` shell syntax: passed
- `prepare_model.sh` shell syntax: passed
- `quantize_selective_gptq.py` Python compile: passed
- `quantize_selective_gptq.py --help` in local SGLang env: passed
- offline quantization run: passed
- SGLang load with `--quantization gptq --dtype float16`: passed
- server startup: passed
- local sparse MiniCPM dtype patch to float16: required for real requests
- ultra-short eval: failed (`0.00%`, mostly empty outputs)
- ultra-short bench: service responsive, but outputs still mostly empty

### Current conclusion

- This path proves that a **competition-valid `prepare_model.sh` offline conversion route is feasible**.
- It does **not** prove that RTN is a viable final quantization method for MiniCPM-SALA.
- The main failure mode is no longer “cannot quantize” or “cannot load”.
- The main failure mode is now “can run but produces empty / useless outputs”.

### Next step

- Stop treating RTN as the main direction.
- Move the next serious exploration to **AWQ-first**, while reusing the already-proven offline `prepare_model.sh` route.
- Keep `v3-fusion-flashinfer` as the stable fallback.

## 2026-03-25 - Deep Investigation: Quantization & Optimization Status

**Status**: All low-risk optimizations already applied, quantization blocked by tooling

### Investigation Summary

**Code-level optimizations checked**:
1. **Fused RMSNorm + residual**: ✓ Already correctly implemented in `minicpm.py`
   - Uses `fused_add_rmsnorm` kernel when residual is provided
   - Week 2 champion recommendation already applied

2. **Rotary embedding FP32 removal**: ✓ Already clean
   - No unnecessary FP32 round-trips in current implementation
   - Cache kept in FP32 for numerical stability (correct behavior)

3. **KV Cache FP8**: ✗ Platform incompatibility
   - `--kv-cache-dtype fp8_e5m2` causes startup failure with flashinfer backend
   - Confirmed in `project-facts.md`

### Weight Quantization Investigation

**Attempted approaches**:
1. **gptqmodel library (v5.8.0)**: ✗ Failed
   - Error: `ModuleNotFoundError: No module named 'transformers.core_model_loading'`
   - Root cause: `defuser` submodule (used by gptqmodel) incompatible with transformers 4.57.1

2. **Transformers + Optimum GPTQConfig**: ✗ Failed
   - Error: `AssertionError: Only flash_attention_2 is supported for sparse attention`
   - Root cause: MiniCPM-SALA custom architecture not compatible with standard HF quantization flow

3. **auto-gptq**: ✗ Failed to build
   - CUDA extension build failure

**Root cause**: MiniCPM-SALA is NOT a standard transformer:
- Hybrid attention (softmax + linear attention)
- Custom sparse attention with top-k selection
- Special InfLLMv2 attention blocks
- Custom modeling code via `auto_map`

**SGLang native quantization**: SGLang supports `--quantization gptq` but expects **pre-quantized weights** in GPTQ format. The quantization must be done offline.

### Promising Direction: Pre-quantized Weights

HuggingFace has quantized MiniCPM-SALA models available. If we can find a GPTQ or AWQ quantized version:
- SGLang can load it directly with `--quantization gptq` or `--quantization awq`
- No need to run quantization ourselves
- Just download and test

**Next step**: Search for pre-quantized MiniCPM-SALA on HuggingFace

### Stable Submission Status

**v3-fusion-flashiefer**: ✓ Protected, untouched

---

## 2026-03-25 - Quantization Exploration (BLOCKED)

**Status**: Weight quantization blocked, code optimizations already in place

**Work done**:
1. **Attempted GPTQ W8A16 quantization**
   - Tried `gptqmodel` library (v5.8.0) → Failed: transformers compatibility issue
   - Tried `transformers + optimum` GPTQConfig → Failed: MiniCPM-SALA custom architecture incompatibility
   - Tried `auto-gptq` → Failed to build (CUDA extension issue)

2. **Investigated code-level optimizations**
   - Fused RMSNorm + residual: Already implemented in current SGLang ✓
   - Rotary embedding FP32 removal: Already clean (no unnecessary conversions) ✓
   - KV Cache FP8: Platform incompatibility with flashinfer backend (confirmed in project-facts.md)

**Blocker Details**:
MiniCPM-SALA has a custom architecture with:
- Hybrid attention (softmax + linear attention)
- Sparse attention patterns with custom top-k selection
- Special InfLLMv2 attention blocks requiring `flash_attention_2`

Standard quantization tools expect standard transformer architectures and fail during model loading.

**Full documentation**: See `submissions/experiments/BLOCKER.md`

**Conclusion**:
- Weight quantization not feasible with current tooling
- Code optimizations (fused RMSNorm, rotary) already in place
- KV cache FP8 blocked by platform flashinfer compatibility
- **v3-fusion-flashinfer remains the best confirmed configuration**

**Files created**:
- `submissions/experiments/BLOCKER.md` (detailed blocker analysis)
- `submissions/experiments/quantize_with_transformers.py` (attempted solution)
- `submissions/experiments/quantize_minicpm_gptq.py` (earlier attempt)

**v3 baseline**: Protected, untouched ✓

---

## 2026-03-25 - Quantization Exploration Setup

---

## 2026-03-25 - Official Platform Result For v3

- submission time: `2026-03-25 11:05:14`
- accuracy: `100.0`
- raw accuracy: `83.78`
- final score: `37.36`
- benchmark duration:
  - `S1 = 864.94`
  - `S8 = 929.87`
  - `Smax = 1398.49`

Interpretation:

- This is currently the best confirmed official result in this repo.
- Compared with the previous official run, all three duration tiers improved.
- Accuracy also improved from `79.44` raw to `83.78` raw, so the final score gain is not only from speed.

## 2026-03-24 / 2026-03-25 - Previous Official Platform Result

- accuracy: `99.31`
- raw accuracy: `79.44`
- final score: `35.16`
- benchmark duration:
  - `S1 = 920.03`
  - `S8 = 987.22`
  - `Smax = 1484.69`

Interpretation:

- Official hidden speed set is much harsher than the earlier local mini-bench.
- Long outputs and long contexts dominate the wall-clock time.
- Compared with the newer official run, this older submission is slower and less accurate.

## 2026-03-25 - Official Result Delta (v3 vs previous official run)

- final score: `+2.20` (`35.16 -> 37.36`)
- raw accuracy: `+4.34` (`79.44 -> 83.78`)
- `S1`: `-55.09 s`
- `S8`: `-57.35 s`
- `Smax`: `-86.20 s`

## 2026-03-25 - Local v3 Fast Eval Smoke Test

- code version: `v3-fusion-flashinfer`
- backend: `flashinfer`
- dataset: `fast_eval_mini.jsonl`
- samples: `10` total, `2` shortest per task
- average score: `97.00%`
- duration: `79.02 s`
- output TPS: `133.37 tok/s`
- result file: `outputs/20260325_113313/predictions.jsonl`

Interpretation:

- This is only a smoke test.
- It is good enough to show the model still behaves reasonably after the `minicpm.py` optimization and backend switch.
- It is not strong enough to certify final competition accuracy.

## 2026-03-25 - Local v3 Fast Bench

- status: `completed`
- default smoke dataset: `bench_speed_v2_smoke.jsonl`
- proxy dataset: `bench_speed_v2_mini.jsonl`
- note: smoke is for quick daily checks, proxy is for pre-submit comparison
- smoke result:
  - `S1 = 116.05`
  - `S8 = 66.32`
  - `Smax = 66.16`
  - wall-clock end-to-end: about `269 s` (`4 min 29 s`)

Interpretation:

- Smoke target is achieved: daily bench now fits under 5 minutes.
- `S1` is faster than the old tiny mini-bench baseline, but `S8` and `Smax` are slower because this smoke set still keeps a more realistic long-context mix.
- Use smoke for quick regression checks and proxy for stronger comparison before submission.
