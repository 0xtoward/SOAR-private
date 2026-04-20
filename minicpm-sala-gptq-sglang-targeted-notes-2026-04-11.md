# MiniCPM-SALA + SGLang + GPTQModel 定向量化决策笔记

日期：2026-04-11

这份笔记是把四类信息压到一起后的项目内决策 memo：

- 当前本地工作区与远端执行环境的真实状态
- 这几天关于 `MiniCPM-SALA + SGLang + GPTQModel` 的已有问答和源码核对
- 你刚提供的那份“2026 年大模型推理量化：精度与速度的工程最佳实践”长回复
- 指定对话 `019d76ad-9a3d-7843-81d3-341cb7e2c719` 的最新几条消息与结果

## TL;DR

- 对我们这条线，`W4A16` 仍然是默认主航道，没有理由先退到更泛的 `RTN-only` 或直接跳去更重的体系迁移。
- 但对 MiniCPM-SALA 来说，当前第一瓶颈不是“理论上 4bit 是否更快”，而是：
  - 量化数值稳定性不干净
  - stop / short-answer 行为仍漂
  - SGLang 运行时仍依赖 dense fallback 和 MiniCPM-SALA 特殊兼容
- 现阶段最值得投入的不是再写一份大而全的量化综述，而是三件事：
  - 保住 `quant env / eval env` 双环境分离
  - 继续做 selective per-module GPTQ，而不是全局大改
  - 把 calibration 和 `damp / failsafe / fallback` 这些数值稳定旋钮真正纳入实验矩阵
- `MiniCPM-O support` 对我们有启发，但它不是稀疏模型方案。它证明的是：`GPTQModel` 上游接受“模型专属 `QModel` 适配”，不是说它已经替我们解决了 SALA 的 sparse / hybrid attention 问题。

## 当前路线的真实状态

- 本地仓库是控制面，远端 `rtx6000-2` 才是实际量化和评测执行面。
- 当前已经明确固定为双环境：
  - quant env：`transformers 5.5.0 + gptqmodel`
  - eval env：`transformers 4.57.1 + sglang`
- 这条分离规则已经被验证是必要条件，不应再合并回单环境。

来源：

- [environment-map.md](/Users/ql/cursor/openbmb/environment-map.md)
- [eval-env-minimal.md](/Users/ql/cursor/openbmb/eval-env-minimal.md)
- [install_minicpm_sala_submitmatch_uv_py310_eval.sh](/Users/ql/cursor/openbmb/scripts/install_minicpm_sala_submitmatch_uv_py310_eval.sh)

## 哪些通用量化结论可以直接套到我们这里

- `W4A16` 仍是工程甜点区间，这一点和通用结论一致。
- calibration 分布强烈影响 stop、EOS、格式 token 和 over-generation，这一点和我们本地现象高度一致。
- selective mixed precision / per-module rollback 比“全模型退回高精度”更符合当前阶段，这一点也和通用结论一致。
- 先做 observability，再做小 sweep，再做更重路线，这个优先级对我们也成立。

## 哪些通用结论不能直接照搬

- Marlin / Machete 的高吞吐叙事主要建立在“量化 kernel 已经进入正确密集 GEMM 路径”的前提上。
- 我们当前实际运行链路仍然是：
  - `gptq_marlin`
  - `flashinfer`
  - `force_dense-minicpm`
  - `o_gate / z_proj` 保持非量化兼容处理
- 所以我们不能把外部论文里“W4A16 理论带宽优势”直接等同于“我们现在 wall-clock 必然更快”。
- 对我们来说，当前 end-to-end 变慢经常不是因为每 token 算得慢，而是因为模型吐得太多，或者短 `mcq` 行为不稳定。

来源：

- [submission-drafts/w4a16-marlin-draft/README.md](/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/README.md)
- [sgl-optimization-log.md](/Users/ql/cursor/openbmb/sgl-optimization-log.md)

## 当前最佳主线

- 当前最强本地候选仍是 `stop-aligned64k-v1` 这条 GPTQ 路线。
- 它的核心价值不是“量化已经完全干净”，而是：
  - 相比更早的 py310 GPTQ 路线，下游行为确实改善了
  - 它保住了当前 submission-safe 的 SGLang 兼容链路
  - 它让后续 selective per-module 修复变得有意义
- 但它还不是 fully closed route，因为：
  - `quant_log.csv` 里仍有 `221` 个 `rtn failsafe`
  - medium rerun 有明显的短 `mcq` 漂移
  - 运行时仍是 dense fallback，不是原生 sparse 路径

关键本地证据：

- [quantization-log.md](/Users/ql/cursor/openbmb/quantization-log.md)
- [submission-drafts/w4a16-marlin-draft/README.md](/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/README.md)
- [submission-drafts/w4a16-marlin-draft/RESULTS.md](/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/RESULTS.md)

## 对 `RTN` / `failsafe` / `fallback` 的项目内解释

- 在我们当前 `gptqmodel 5.8.0` wheel 里，这套机制的名字还是 `FailSafe`。
- 在上游较新的 `main` 分支里，这套命名已经更偏 `fallback`，但本质是一回事。
- `RTN` 不是“项目选择的主方法”，而是兜底。
- 它有两条主要触发路径：
  - 低样本模块的主动 fallback
  - Hessian 数值失败后的被动 fallback
- 我们当前 MiniCPM-SALA 路线主要命中的是第二类，也就是：
  - Cholesky 不稳定
  - damp recovery 失败
  - diagonal floor 也救不回来
  - 然后退到 `rtn failsafe`

这点很关键，因为它决定了下一步不能只说“多给点 calibration 就好了”，而要同时处理：

- calibration 语义质量
- calibration 分布相关性
- per-module selective protection
- `damp_percent / damp_auto_increment / failsafe smoothers`

当前 `GPTQModel` 已知可用的缓解层包括：

- 默认 `failsafe/fallback` 阈值
- `damp_percent`
- `damp_auto_increment`
- Hessian diagonal floor
- fallback smoothing
- per-module dynamic quantization / exclusion

但我们自己的量化脚本目前真正暴露出来、已经能直接用的主要是：

- `group_size`
- `desc_act`
- `--dynamic-config-path`
- per-module `dynamic` manifest 落盘

还没有直接暴露出来的关键旋钮是：

- `failsafe` / `fallback` 策略
- `damp_percent`
- `damp_auto_increment`
- fallback smoother 配置

这意味着：如果下一轮要认真打 `RTN` warning 缓解，脚本层还需要一小步补参。

来源：

- [scripts/quantize_gptq_w4a16.py](/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py)
- [quantization-log.md](/Users/ql/cursor/openbmb/quantization-log.md)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)

## 为什么 `MiniCPM-O support` 不能当作 SALA 已有解

- `MiniCPM-O 4.5` 官方模型卡写的是：
  - 基于 `SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B`
  - 总计约 `9B`
  - 模态模块与 LLM 主干是“densely connected”
- 官方 `configuration_minicpmo.py` 里主配置类继承的是 `Qwen3Config`，不是 MoE 配置。
- `GPTQModel` 对 `MiniCPM-O` 的适配重点是：
  - remote code / cache API 兼容
  - processor 输入整理
  - `generate()` 式 input capture
  - 视觉 / 音频 / TTS 模块的 materialize / offload
- 它不是 sparse attention / expert routing / heterogeneous layer tree 解决方案。

对我们真正有用的洞见只有两个：

- `GPTQModel` 上游接受模型专属 `QModel` 适配
- 复杂性应该尽量收口在单模型 definition 里，而不是污染全局量化框架

这和 `MiniCPM-SALA` 很像的地方在“上游工程风格”，不像的地方在“问题类型”。

我们的主要问题仍然是：

- hybrid attention
- `o_gate / z_proj`
- SGLang 侧 checkpoint / runtime 兼容
- 某些投影层量化后行为明显更脆

来源：

- [MiniCPM-o 4.5 model card](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [configuration_minicpmo.py](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/configuration_minicpmo.py)
- [GPTQModel minicpm_o.py](https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/models/definitions/minicpm_o.py)

## 从最新 `019d76ad...` 对话里得到的直接行动结论

这条对话的最新状态不是“抽象计划”，而是已经跑到了 selective candidate 的第一轮证伪。

已确认的事实：

- selective 计划的 Wave 1 核心思路是对 `o_proj / down_proj` 做最小 selective 修复。
- 当前工作区里的量化脚本已经带有 `--dynamic-config-path` 与 dynamic manifest，不再只是想法。
- Candidate A `odown-gs64` 已经在 `focus10` 上被直接淘汰：
  - `167 / 589` 两个关键 `mcq` 没救回来
  - `344` 也翻错
  - `31433 cwe` 从健康短尾炸到 `65199`
  - 最终 `focus10` 只有 `49.00%`
- 最新看到的后续动作是：
  - Candidate B `skip-o-down64` 已经启动
  - 当时量化在进行中，但我还没有在这次整理里看到它的最终评测结果

这说明一个非常重要的项目内结论：

- “把坏层的 `group_size` 从 128 改到 64”不是天然安全的小修补
- 至少在 `o_proj / down_proj` 这一组上，单纯 `gs64` 可能会让短 `mcq` 更不稳定
- 下一步 selective 设计应该优先允许：
  - 极少量模块直接排除不量化
  - 或更明确地用 dynamic rule 做更保守的 per-module rollback

来源：

- [session 019d76ad-9a3d-7843-81d3-341cb7e2c719](</Users/ql/.codex/sessions/2026/04/10/rollout-2026-04-10T17-16-22-019d76ad-9a3d-7843-81d3-341cb7e2c719.jsonl>)
- [worklog.md](/Users/ql/cursor/openbmb/worklog.md)
- [quantization-log.md](/Users/ql/cursor/openbmb/quantization-log.md)

## 当前最值得执行的优先级

### P0：保持环境和观测面稳定

- 不要再把 quant / eval 环境混回一起。
- 评测时固定同时记录：
  - score
  - output token count
  - finish / truncation 行为
  - per-row 长尾
- 对我们来说，`TPS` 单独看没有决策意义。

### P1：继续 selective GPTQ，但只做 submission-safe 的最小动作

- 优先动作：
  - 极少量层排除不量化
  - 少量层更保守 `group_size`
- 暂时不把 mixed-bit Marlin 当第一波默认方案，因为当前 serving 兼容性还没有在这条链上闭环。
- Wave 1 里 `Candidate B / C / D` 这类 selective 方案仍值得继续，但必须用 `focus10` 先卡掉。

### P2：把 calibration 从“长度覆盖”推进到“数值稳定友好”

- 当前 stop-aligned64k 的价值已经被证明，但它依然数值激进。
- 认真版本应转向：
  - `128` 到 `256+` 条 coherent 样本
  - 真实 prompt 结构
  - chat-close / stop token 周边状态保留
  - 少一些巨大 synthetic long row 的支配

### P3：补脚本参数，把 `RTN` 缓解变成可实验对象

- 建议补到 CLI 的旋钮：
  - `--damp-percent`
  - `--damp-auto-increment`
  - `--failsafe-json` 或等价配置入口
- 当前如果不补这层，我们实际上只能在：
  - calibration
  - `group_size`
  - `desc_act`
  - `dynamic` exclusion / override
  这几项里绕圈。

### P4：重路线只放在后手

- `AWQ` 仍有研究价值，但它现在更像 post-GPTQ fallback route，不是立刻替换主线。
- `OmniQuant`、`W4A8KV4`、更激进 KV cache 量化，都不是当前最省 GPU 时间的下一步。
- `MiniCPM-O` 上游适配经验值得留档，但不要把它误判成 SALA 的当前 blocker 解法。

## 我会如何定义“接下来一轮实验是否成功”

- 不是只看 `TPS` 是否上升。
- 也不是只看总 token 是否下降。
- 对当前 MiniCPM-SALA 路线，更合理的成功标准是：
  - `focus10` 先过
  - `167 / 589` 至少修一个
  - 不把健康短尾 `cwe` 拉炸
  - full medium 不出现广泛回退
  - `rtn failsafe` 模块数下降，或者至少坏层集中度更可解释

## 当前不建议做的事情

- 不建议为了“理论先进性”先迁去更重的系统协同路线。
- 不建议把 `MiniCPM-O support` 当作 SALA upstream-ready 的证据。
- 不建议再花时间做 blanket runtime penalty sweep。
- 不建议只因为外部论文里 `W4A16 + Marlin` 很快，就忽视我们当前 dense fallback 和 output explosion 的真实瓶颈。
- 不建议把 `221` 个 `rtn failsafe` 当作“反正能跑就行”的小 warning。

## 最后一句判断

如果只用一句话概括现在这条线：

`W4A16 + GPTQModel + SGLang` 仍然是 MiniCPM-SALA 当前最值得继续深挖的主线，但它已经进入“选择性修坏层 + 修 calibration 数值稳定 + 保住 runtime 兼容”的阶段，而不是“再证明一次 4bit 值不值得”的阶段。

## 参考源

本地：

- [environment-map.md](/Users/ql/cursor/openbmb/environment-map.md)
- [eval-env-minimal.md](/Users/ql/cursor/openbmb/eval-env-minimal.md)
- [quantization-log.md](/Users/ql/cursor/openbmb/quantization-log.md)
- [sgl-optimization-log.md](/Users/ql/cursor/openbmb/sgl-optimization-log.md)
- [worklog.md](/Users/ql/cursor/openbmb/worklog.md)
- [calibration-handoff.md](/Users/ql/cursor/openbmb/calibration-handoff.md)
- [model-notes.md](/Users/ql/cursor/openbmb/soar-docs/my_doc/model-notes.md)
- [w4a16-marlin-draft README](</Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/README.md>)
- [quantize_gptq_w4a16.py](/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py)
- [session 019d76ad...](</Users/ql/.codex/sessions/2026/04/10/rollout-2026-04-10T17-16-22-019d76ad-9a3d-7843-81d3-341cb7e2c719.jsonl>)

外部：

- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [SGLang](https://github.com/sgl-project/sglang)
- [MiniCPM-o 4.5 model card](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [MiniCPM-o 4.5 configuration](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/configuration_minicpmo.py)
