---
title: "Infinite Worlds with Versatile Interactions"
authors: "Zelin Gao et al."
arxiv: "2607.07534"
url: "https://arxiv.org/abs/2607.07534"
year: 2026
date_read: "2026-07-15"
venue: "arXiv preprint"
category: "world-model"
tags: [world-model, video-generation, action-control, diffusion-models, transformer]
rating: 4
code: "https://github.com/robbyant/lingbot-world-v2"
status: read
---

# Infinite Worlds with Versatile Interactions

> 一句话总结（TL;DR）：论文提出 LingBot-World-Infinity，通过 MoBA 掩码的因果预训练与后续少步蒸馏，实现高交互、多动作、长时稳定的实时世界模型生成。

## 我的问答记录（QA）

### Q1: 它提出的那个 attention mask 机制是什么意思，我没有理解，能详细说说吗
MoBA（Mixture of Bidirectional and Autoregressive）本质是在因果生成约束里加入一个双向块。论文在 Sec. 3.2 和 Fig. 4 里给出结构：一部分自注意力保持 teacher-forcing 的因果可见性（只能依赖历史），另一部分追加 bidirectional full-attention 区域，作为正则化以缓解纯因果训练在长上下文下的过度依赖历史问题。配套的 cross-attention 也按两路设计：自回归路用背景+历史 chunk prompt 的下三角可见性，双向路接全局 prompt，避免 future semantics 泄漏。

### Q2: 我看他图上画了横向是 x0x1x2; x0^t x1^t x2^t; x0^t x1^t x2^t，这些 x0x1x2 分别是什么意思，为什么重复了三次
这里有两个时间轴：下标 i（0/1/2）是视频帧顺序，上标 t 是扩散噪声时刻。x_i 是干净帧 latent，x_i^t 是该帧在噪声时刻 t 的状态（与 Eq. (2) 对应的 flow-matching 训练变量一致）。图里“重复三段”不是三份不同视频，而是为了在同一批输入上施加两种掩码行为：一段 clean context + 两段 noisy 分支（AR 分支与 Bi 分支），这样同一内容可分别执行因果约束和双向约束。

### Q3: 这个 attention mask 对应实际 inference 的时候怎么做？看起来 context 增长了 3 倍
论文未给出逐算子级实现细节，但从 Sec. 3.2 与 Sec. 4.3.2（Dynamic KV Cache Management）可以确定：部署时仍是因果自回归生成，不会允许未来帧泄漏。图中的“三段拼接”主要是训练期的可见性构造；推理时的主要增长来自常规历史 KV 累积，而不是简单把长期上下文永久复制三份。并且他们在部署侧引入动态 KV 调度来压制长时成本。

### Q4: 你意思是训练的时候才是这么训练的，推理的时候就是一个正常的 causal mask 对吗
更准确说法是：训练阶段明确使用 MoBA 混合掩码（Sec. 3.2）；推理阶段必须满足因果自回归约束。论文没有把“是否保留训练图里的显式双分支形式”写成实现细节，因此能确定的是因果性不变，不能确定的是具体工程上是否采用最朴素单一路 textbook causal mask。

### Q5: 能讲细节一点吗，从 video clip 处理到 token 到 self-attention 之前，重点讲怎么构造加噪数据
核心加噪变量在 Eq. (2)：对目标帧 latent 构造 x_i^t=(1-t)x_i+t\epsilon，t~U(0,1), \epsilon~N(0,I)，并让网络预测速度项 (\epsilon-x_i)。在 MoBA 下，进入 self-attention 前不是只喂一份 noisy token，而是组织为 clean context + noisy(AR) + noisy(Bi) 的混合输入，并用对应的自注意力/交叉注意力掩码分别约束。也就是说，加噪公式本身是 flow-matching 标准形态，MoBA 创新在“加噪后如何分支并施加混合可见性”。

### Q6: 绝对不是这个意思，你说的是标准 video gen 流程，完全没有包括 MoBA 相关技术
澄清后，讨论聚焦为：MoBA 的关键不是替代扩散加噪公式，而是把同一 noisy 样本复制为 AR 与 Bi 两路查询分支，再配合块状 mask 同时训练“因果可生成性 + 双向表示能力”，并在 cross-attention 上使用“下三角 chunk prompt + 全局 prompt”双路策略（Fig. 4, Sec. 3.2）。

## 我的收获 / 结论
- 这篇里 MoBA 最容易误读的点是“图里的三段 token 拼接”与“部署时真实 KV 缓存布局”不是一回事。前者是训练期可见性构造，后者是推理期系统实现问题。
- 从理解上，抓住两件事就够：
  1) Eq. (2) 的加噪与流匹配目标；
  2) Fig. 4 的 AR/Bi 混合掩码与配套 cross-attention 防泄漏设计。

## 关联工作 / 后续可读（可选）
- Consistency Models（对应 Sec. 3.3 的 few-step consistency distillation）
- DMD / One-step Diffusion Distillation（对应 Sec. 3.3 的分布匹配蒸馏）
- Self Forcing（论文引用用于缓解 train-test gap 的长滚动优化）

## 关键摘抄（可选）
- “Mixture of Bidirectional and Autoregressive Attention Mask” 的核心描述见 Sec. 3.2 / Fig. 4。
- Eq. (2): 条件 flow-matching 目标，速度监督为 (\epsilon - x_i)。
- Sec. 4.3.2: 部署侧 Dynamic KV Cache Management，强调长时推理效率与相关历史选择。