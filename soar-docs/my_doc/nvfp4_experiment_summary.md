# NVFP4 量化实验总结

## 实验状态

**日期**: 2026-03-28
**模型**: MiniCPM-SALA (9.48B 参数, ~18GB BF16)
**目标**: W4A16 NVFP4 量化

## 量化结果

| 指标 | 值 |
|------|-----|
| 原始大小 | ~18 GB |
| 量化后大小 | **6.20 GB** |
| 压缩比 | ~3x |
| 量化工具 | modelopt 0.42.0 |
| 校准数据 | calibration_data_merged.jsonl (128 samples) |

## 兼容性问题分析

### 问题1: 空Tensor崩溃 (已修复)

**错误位置**: `minicpm_backend.py:448`
```
RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0
```

**原因**: NVFP4量化后，某些sparse attention分支的tensor为空，调用`.max()`崩溃。

**修复**: 添加空tensor检查
```python
if seqlen_q_sparse_tensor.numel() > 0:
    metadata.max_seqlen_q_adjusted = seqlen_q_sparse_tensor.max().item() * self.heads_per_group
else:
    metadata.max_seqlen_q_adjusted = 0
```

### 问题2: Sparse K1 Cache写入失败 (未解决)

**错误位置**: `memory_pool.py:635`
```
RuntimeError: The expanded size of the tensor (511) must match the existing size (0)
```

**原因**: MiniCPM-SALA的sparse attention机制与NVFP4权重量化格式存在深层不兼容：
1. NVFP4使用block-wise量化(group_size=16)
2. MiniCPM-SALA使用混合注意力(Lightning Attention + MiniCPM4)
3. Sparse TopK检索时，量化后的权重scale与原始sparse cache索引不匹配

### 根本原因

NVFP4是**实验性格式**（SGLang日志明确提示），专为Blackwell GPU设计，但：
1. DeepGemm需要`ue8m0` scale格式，当前checkpoint不是
2. MiniCPM-SALA的自定义sparse attention未被modelopt完整支持
3. KV cache量化与sparse TopK行为冲突

## 下一步: W4A16 GPTQ + Marlin

Week 4冠军路线推荐：
- **W4A16 + GPTQ + Marlin** 是成熟稳定的方案
- 已在v7-awq-aggressive验证过可行性
- Marlin kernel对Blackwell有良好支持

## 文件位置

| 文件 | 路径 |
|------|------|
| 量化模型 | `/root/autodl-tmp/models-nvfp4/` |
| 量化脚本 | `/root/autodl-tmp/SOAR-Toolkit/submissions/experiments/quantize_modelopt_nvfp4.py` |
| 修复的后端 | `/root/autodl-tmp/sglang/python/sglang/srt/layers/attention/minicpm_backend.py` |