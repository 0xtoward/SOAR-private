# Week 2 — 曹议：混合精度量化 + AI 工作流（卫冕，达成度 96%）

## 模型代码优化
1. **LayerNorm + 残差融合**：复用 SGLang 现成融合算子
2. **去除 rotary_emb FP32 转换**：原代码 q,k 转 FP32 做 rotary_emb 再转回，去掉减少开销

## 混合精度量化
- KV Cache FP8：精度几乎不掉，免费吞吐提升
- 权重 W4：大幅提速但精度掉
- 策略：敏感层高精度 + 其余 W4

## AI 协同工作流
- Gemini 3.1 Pro：选方向、Deep Research、润色 Prompt
- Claude Code (Opus 4.6)：代码落地、测试执行
- 流程：Gemini 出方案 → 润色 Prompt → Claude Code 改码+测试 → 闭环

## 给 Claude Code 的 Prompt 关键点
- 测试 6 种组合：(长输入短输出, 短输入长输出) × (BS=1, 8, 64)
- 约束：不破坏现有功能，保持 disable_radix_cache
- 工作流：先问后做，得到同意再行动，每次记录结果
