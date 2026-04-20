# Week 5 — 智算一队：NVFP4 KV Cache 与混合精度实践（周冠军）

## 总体路线

第五周冠军路线的重点，不再是重新发明一条新的权重量化主线，而是在已经完成：

- `W4A16 + GPTQ + Marlin`
- `FP8 KV Cache`

的基础上，继续压缩 decode 阶段的 KV Cache 带宽成本。

他们最终押注的是：

1. 先尝试纯 `NVFP4 KV Cache`
2. 发现纯 `FP4` 精度掉得太多
3. 再做逐层敏感性分析
4. 用少量关键层保留 `FP8 KV Cache`
5. 其余层改成 `NVFP4 KV Cache`

一句话：

> 第五周的冠军，不是“全模型改成 NVFP4”，而是“在 W4A16 主线之上，对 KV Cache 做混合精度压缩”。

## 1. 为什么盯住 KV Cache

- decode 阶段更偏 `memory-bound`
- 长序列下，KV Cache 占用和读带宽都很重
- 如果能把 `FP8 KV` 再压到 `FP4 KV`，理论上又能再减半一部分 KV 存储和带宽

这和第四周 `W4A16 + GPTQ + Marlin` 的思路是一致的：

- 第四周压的是权重带宽
- 第五周继续压的是 decode 时的 KV Cache 带宽

## 2. NVFP4 KV Cache 是什么

- `NVFP4` 使用 `FP4 E2M1`
- 两个 `FP4` 值打包进一个 `uint8`
- 每个量化 block 配一个 scale factor
- 反量化时乘上 scale 恢复近似值

对 KV Cache 来说，这样做的意义是：

- Cache 更小
- decode 读取带宽更低
- 长上下文时收益更明显

## 3. 纯 FP4 为什么不够

他们明确说了：

- 纯 `FP8 KV Cache` 正确率大约还能保持在 `80%`
- 纯 `NVFP4 KV Cache` 会掉到大约 `75%`

也就是说：

> 纯低比特不是不能跑，而是精度代价太大，不能直接拿来当最终方案。

## 4. 冠军真正的关键：混合精度 KV Cache

他们不是“全层统一降到 FP4”，而是做了逐层敏感性分析：

- 逐层把某一层 KV Cache 从 `FP8` 降到 `FP4`
- 观察精度下降幅度
- 用下降幅度判断该层是否敏感

最后得到的结论是：

- 首层和末尾几层最敏感
- 中间层更鲁棒

于是最终方案变成：

- 少量关键层保留 `FP8 KV`
- 大部分中间层使用 `NVFP4 KV`

这个结果很重要，因为它说明：

> 对长序列模型来说，“少数关键层保高精度，其余层压低比特”往往比“全层统一降比特”更有效。

## 5. 他们到底做了哪些工程工作

这篇笔记里提到，他们不是只改了一个量化开关，而是把整条链路都打通了：

- `FP4` 内存池管理
- attention backend 适配
- KV 写入量化路径
- KV 读取反量化路径

所以这个冠军方案的本质不是“用了 NVFP4”这么简单，而是：

> 做了完整的、可运行的 NVFP4 KV Cache 系统实现，再用混合精度把精度拉回来。

## 6. 和我们最相关的启发

### 启发 1：他们的 NVFP4 不是我们的 NVFP4

他们这周说的核心是：

- `NVFP4 KV Cache`

而我们之前主要尝试的是：

- `ModelOpt NVFP4` 权重量化

这是两条不同路线，不能直接类比成“别人 NVFP4 成了，我们的 NVFP4 也应该只差一点参数”。

### 启发 2：混合精度比一刀切更现实

如果低比特直接掉精度，不应该只在“全开”和“全关”之间切换。

更好的思路是：

- 找敏感层
- 给敏感层保更高精度
- 其余层再压低比特

### 启发 3：离线校准和层级敏感性分析同样关键

冠军明确提到：

- 低比特不是简单开关
- 需要离线校准
- 需要逐层敏感性分析

这和我们最近在 `GPTQ stop-aligned calibration`、`selective layer treatment` 上得到的经验是相通的。

## 对我们当前路线的直接含义

1. 不要把“冠军用了 NVFP4”误读成“全模型 NVFP4 权重量化已经被证明适合 MiniCPM-SALA”
2. 如果再回头看 `NVFP4`，优先看的应是：
   - `KV Cache` 路线
   - 混合精度路线
   - 逐层敏感性路线
3. 如果继续做 `GPTQ/Marlin`，也可以借鉴他们的核心方法论：
   - 先找敏感层
   - 再做 selective / mixed precision
   - 不要默认全层统一改动一定更优

## 7. 我们自己的收获（2026-04-12 补充）

这周冠军笔记和我们最近几天的实验对照后，我认为有几条收获必须单独记下来。

### 收获 1：这篇笔记强化了 Week 4，不是否定 Week 4

第五周并不是“放弃 `W4A16 + GPTQ + Marlin`，转向全新主线”。

更准确地说：

- Week 4 先把主权重路线做对
- Week 5 再在 decode 阶段追加 `KV Cache` 压缩

所以对我们来说，当前主线仍然应该优先是：

- `W4A16 + GPTQ + Marlin`

而不是把注意力又拉回“全模型 `NVFP4` 权重量化”。

### 收获 2：我们失败的 NVFP4 路线，并不是冠军笔记里的那条路

我们踩坑最多的是：

- `ModelOpt NVFP4` 权重量化

而冠军这周真正做成的是：

- `NVFP4 KV Cache`
- 并且是混合精度 `KV Cache`

我们自己的日志已经证明，之前那条 `modelopt NVFP4` 权重量化路线的问题很深：

- `MiniCPM-SALA` 自定义 sparse attention 支持不完整
- `DeepGemm` 与当前 checkpoint `scale_fmt` 不匹配
- `bf16 export -> fp16 serve` 还引入了额外不一致
- 后续甚至直接触发了 `CUDA illegal memory access`

所以现在不能把“冠军用了 NVFP4”理解成“我们之前失败的 NVFP4 只差一点小调参”。

### 收获 3：我们最近的 selective `C` 其实违背了一个更稳妥的原则

冠军第五周强调的是：

- 对敏感部分保高精度
- 对相对不敏感部分降比特

而我们最近的 selective `C` 做的是：

- 直接把一部分 attention `q/k/v/o` 改成 `group_size=64`

这件事虽然工程上被我们修到可加载了，但结果上：

- 官方 `acc` 从之前较好的 `95.78~95.89`
- 掉到 `94.94`

这说明：

> 在我们当前这条 MiniCPM-SALA 路线上，attention 侧 selective 改动的风险，暂时高于它的收益。

### 收获 4：校准和混合精度都重要，但方向必须和真正瓶颈对齐

我们最近从 `stop-aligned calibration` 得到的主要收益，不是 kernel 变快，而是：

- over-generation 下降
- 输出 token 变少
- 局部 stop 边界变稳

这件事和冠军笔记强调的“离线校准 + 敏感层保真”是相通的。

但我们也看到：

- `stop-aligned` 还没有稳定转化成官方 `97+ acc`
- `221` 个 `rtn failsafe` 说明当前 GPTQ 数值仍不够干净

所以接下来不能只靠“再调一点 runtime penalty”或者“再多动一点 attention gs64”，而要把校准、敏感层保护、以及官方评分门槛一起看。

### 收获 5：如果重看 NVFP4，应该把它当成“新路线”，不是旧失败路线的小修补

如果后面真的要重启 `NVFP4`，更合理的方向是：

- `KV Cache`
- 混合精度
- 逐层敏感性分析

而不是继续在我们之前那条：

- `ModelOpt NVFP4` 权重量化

的 submission scaffold 上做小修小补。
