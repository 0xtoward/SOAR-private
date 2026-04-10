# 冠军三周技术路线演进

```
Week 1: 粗暴 W4 → 91分 (C=0.92, perf=98.91)
Week 2: 混合精度 + 代码优化 + KV FP8 → 卫冕 (达成度 96%)
Week 3: NVFP4 + 校准数据优化 + logprobs 验证工具 → 三连冠 (达成度 97%)
```

## 冠军用到的所有技术

| 技术 | 周次 | 效果 |
|---|---|---|
| W4 权重量化 | W1 | S1 降 40% 耗时，但 C=0.92 |
| KV Cache FP8 | W1-W2 | 免费吞吐提升，精度几乎不掉 |
| LayerNorm+残差融合 | W2 | 减少 kernel launch |
| 去除 rotary_emb FP32 转换 | W2 | 减少 dtype 转换开销 |
| 混合精度（敏感层高精度） | W2 | 兼顾速度和精度 |
| NVFP4 (E2M1) | W3 | Blackwell 原生加速 |
| logprobs 精度验证 | W3 | 提交前快速筛方案 |
| 校准数据优化 | W3 | 降低随机词比例，提升精度稳定性 |

## 评测机硬件确认
- **Blackwell 架构**（RTX PRO 6000 Blackwell, ~96GB）
- NVFP4 有原生硬件加速
- 平台 flashinfer 版本不支持 FP8 KV Cache（fa2 backend 限制）
