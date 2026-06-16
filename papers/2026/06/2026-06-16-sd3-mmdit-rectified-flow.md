---
title: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
authors: "Patrick Esser, Sumith Kulal, Andreas Blattmann, Robin Rombach et al. (Stability AI)"
arxiv: "2403.03206"
url: "https://arxiv.org/abs/2403.03206"
year: 2024
date_read: "2026-06-16"
venue: "ICML 2024"
category: "image-generation"
tags: [diffusion-models, rectified-flow, mm-dit, text-to-image, scaling, transformer]
rating: 5
code: "https://github.com/Stability-AI/sd3-ref"
status: read
---

# Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

> 一句话总结（TL;DR）：这就是 Stable Diffusion 3 背后的论文——通过把 rectified flow 训练的时间步采样偏向感知上最 relevant 的中间尺度（logit-normal 采样），并提出文本/图像双流双向交互的 MM-DiT 架构，把「RF + Transformer」扩展到 8B 参数，得到在提示遵循与人类偏好上超越 SDXL、DALL·E 3 的高分辨率文生图模型，且 scaling 趋势可预测无饱和。

## 我的问答记录（QA）
> 本节是核心：按阅读时的真实对话顺序，记录我提出的问题和对应解答。

### Q1: MM-DiT 架构现在已经成为一个非常通用的结构，这个架构具体指的是什么，其特点是什么？

MM-DiT（Multimodal Diffusion Transformer，第 4 节 + Figure 2）建立在 DiT 之上，针对「文本 + 图像」两模态改造。

**核心思想**：为文本和图像两种模态使用**两套独立的权重**，但在注意力运算时把两个模态的序列**拼接到一起做联合注意力**。论文原文：等价于「为每个模态各设一个独立的 Transformer，但在 attention 操作时把两个模态的序列 join 起来，从而两种表示既能在各自空间里工作，又能彼此参考对方」。

**输入与条件化**：
- 图像分支：在自编码器 latent 空间，把 latent 切 2×2 patch、加位置编码、flatten 成长度 $\frac12 h \cdot \frac12 w$ 的序列。
- 文本分支：pooled 表示（粗粒度，CLIP-L/14 + CLIP-G/14）和 timestep 嵌入一起通过 modulation 注入；序列表示 $c_{ctxt}$（细粒度，CLIP + T5-XXL）作为 token 与图像 token 拼接进注意力。仅用 pooled 不够（只保留 coarse-grained 信息）。

**关键特点**：
1. **双流 + 双向信息流**：不同于传统 cross-attention 单向喂文本，MM-DiT 让文本/图像 token 在 joint attention 里双向互相影响，改善文本理解、排版、人类偏好。
2. **模态特定权重**：两模态概念差异大，各用一套 QKV/LN/MLP。试过三套（CLIP/T5 再拆）但只有 small gain，最终用两套（5.2.3 节）。
3. **DiT 风格 modulation 条件化**：用 timestep $t$ 和 pooled 文本向量 $c_{vec}$ 生成每个 block 的调制参数（$\alpha,\beta,\gamma,\delta,\epsilon,\zeta$）。
4. **可选 QK-RMSNorm** 稳定训练。
5. **按 depth 缩放**：hidden = $64d$，MLP = $4\cdot64d$，heads = $d$；据此扩到 8B（$d=38$）。

对比（5.2.3 节 / Figure 4）：vanilla DiT < UViT < CrossDiT < **MM-DiT**。DiT 可看作 MM-DiT「所有模态共享一套权重」的特例。

### Q2: 能不能写几个 block 的 PyTorch 示例代码？

写了一份贴着 Figure 2b 的最小实现，核心要点：
- `mod_c/attn_c/mlp_c` 与 `mod_x/attn_x/mlp_x` 完全分离 = 两套独立权重（"two independent transformers"）；共享一套就退化成普通 DiT。
- 灵魂在 joint attention：`q = torch.cat([qc, qx], dim=2)`、`k/v` 同理，拼接后做**一次** `scaled_dot_product_attention`，文本 query 能对图像 key 打分、反之亦然 → 双向流；算完再按原长度拆回两条流各走输出投影。
- modulation：6 组参数里 shift/scale 调制 LN 输出，gate（γ、ζ）做残差门控，全来自条件向量 `y`（DiT 的 adaLN-Zero 风格）。
- QK-RMSNorm 对应 5.3.2 节。
- 整体由 depth `d` 决定：`dim = 64*d`，`n_heads = d`，`depth` 个 block 串联，两条流并行往下传。

### Q3: 这篇在 diffusion 调参里很出名，它测了很多 loss 组合，最终结论是什么？能给 diffusion training 的调参 hint 吗？

第 5.1 节：在 ImageNet / CC12M 上训了 **61 种 formulation**（loss + 噪声调度 + 时间步采样），用 24 种控制设定做 non-dominated sorting 排名（Table 1/2，Figure 3）。

**赢家：`rf/lognorm(0.00, 1.00)`** —— rectified flow + logit-normal 时间步采样（$m=0, s=1$）。全局排名第 1（rank 1.54），5 步和 50 步都稳居前列，跨指标/跨数据集都好、无短板。

**实验说明了什么**：
1. **中间时间步更重要 → 要对 timestep 加权采样**（3.1 节）。RF 目标 $\epsilon-x_0$ 在 $t\to0/1$ 时简单（最优解是某分布均值），难点全在中间。logit-normal 峰值在 $t=0.5$，明显优于 uniform（`rf`，rank 5.67）。
2. **RF 在少步采样（5~10 步）优势最大**（Figure 3）；步数 ≥25 时只剩 `rf/lognorm(0,1)` 能跟 `eps/linear`（LDM-Linear）掰手腕。
3. **只有「带改进时间步采样的 RF」才打得过老牌 LDM-Linear**；朴素 uniform-RF 反而更差。配方（时间步采样）才是关键。
4. **别用单一指标选 formulation**：很多变体单看 CLIP 或 FID 很好但另一指标崩，要看跨指标/跨设定稳定性。

**调参 hint**：
- 首选 RF + logit-normal（$m=0, s=1$，即采 $u\sim\mathcal N(0,1)$ 过 sigmoid）——开箱默认值，SD3 实际所用。
- 核心原则：采样权重往**中间时间步**倾斜，别 uniform。
- 要少步快采 → RF 收益最大；步数充足时 LDM-`eps/linear` 仍是强 baseline。
- 评估要多指标多设定做 Pareto 排序，避免假最优。
- （推断）高分辨率/大模型再叠加 resolution-dependent shift（见 Q4）。

一句话：「RF 不是因为是直线就赢，而是因为可以自由重新加权时间步——把权重压到感知上最 relevant 的中间尺度，才是真正起作用的旋钮。」

### Q4: 高分辨率/大模型的 timestep shift 具体做法是什么？有什么好处/道理？

第 5.3.2 节「Resolution-dependent shifting」。

**直觉**：分辨率越高、像素越多，要摧毁信号需要更多噪声。同一个 $t$ 在高分辨率下噪声「不够狠」，直接照搬低分辨率调度会让时间步与真实信噪难度错位。

**推导（常数图像思想实验）**：设 $n=H\cdot W$ 像素，常数图像前向 $z_t=(1-t)c\mathbf1+t\epsilon$，每个像素是同一随机变量 $Y=(1-t)c+t\eta$ 的观测。$\mathbb E(Y)=(1-t)c$，$\sigma(Y)=t$。用样本均值估 $c$，估计的标准误差：
$$\sigma(t,n)=\frac{t}{1-t}\sqrt{\frac1n}$$
$n$ 以 $1/\sqrt n$ 出现 → 宽高都翻倍（$n\times4$）则同一 $t$ 的不确定性减半，正式说明高分辨率下噪声不够。

**做法**：让两分辨率在「不确定性相等」下对齐时间步，$\sigma(t_n,n)=\sigma(t_m,m)$ 解出：
$$t_m=\frac{\sqrt{m/n}\,t_n}{1+(\sqrt{m/n}-1)t_n}\quad(\text{Eq. 23})$$
定义 shift $\alpha:=\sqrt{m/n}$，等价于 log-SNR 常数平移 $\lambda_{t_m}=\lambda_{t_n}-2\log\alpha=\lambda_{t_n}-\log\frac mn$（Eq. 24–25）。

**实际取值**：常数图像假设不现实，所以用人类偏好实验选 $\alpha$（Figure 6）：shift > 1.5 明显更受偏好，更高值间差别不大。最终 1024×1024 训练 + 采样都用 **$\alpha=3.0$**。

**好处/道理**：① 对齐时间步与真实感知难度；② 修正高分辨率噪声不足，把监督推向真正困难的噪声水平；③ 训练和采样双端都用，质量明显提升；④ 单参数 $\alpha=\sqrt{m/n}$，理论简洁、等价 log-SNR 常数平移、易手调。与「中间时间步更重要」是同一思想（让时间步分布对齐感知 relevant 噪声尺度）。

### Q5: QK-Norm 那个图表（Figure 5）是为了说明什么？RMSNorm 是什么，是本文首次提出的吗？公式是什么？

**Figure 5（5.3.2 节）**：取 2B（$d=24$）模型**最后 5 个 block 的平均**。左图 = 最大 attention logit 随训练变化；右图 = attention 熵。不加 QK-Norm 时 logit 不受控疯涨（attention-logit growth instability），熵随之塌缩（注意力退化成近 one-hot），训练发散。加 QK-Norm 后两者都稳住。结论：① 高分辨率 + bf16-mixed 下存在不稳定性；② QK-Norm 比退回 full precision（约 2× 性能损失）更高效，能在 bf16-mixed 稳定训练（配 AdamW $\epsilon=10^{-15}$）；③ 与判别式 ViT 文献不同，本文观察到不稳定**集中在最后几个 block**。

**RMSNorm 不是本文首创**，引用 Zhang & Sennrich, 2019；「对 Q/K 归一化稳训练」来自 Dehghani et al. 2023。本文只是把它用作 QK-Norm 算子并验证。

**公式**：
$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\varepsilon}}\odot g$$
$g$ 为可学习缩放（本文用 learnable scale）。与 LayerNorm 区别：**不减均值（不中心化）、无 bias**，只用 RMS 缩放。本文作用在每个 head 的 Q、K 上（归一化维度 = `head_dim`），算 $QK^\top$ 前；两条流都加，标为可选。

直觉：$q\cdot k=\|q\|\|k\|\cos\theta$，RMSNorm 把模长钉死后 logit 只能靠方向变化，掐断「logit 大→梯度大→logit 更大」的正反馈。

### Q6: RMSNorm 有可学习的 $g$ 参数，所谓「限定在固定尺度」怎么理解，大概多大？为什么不用 LayerNorm？

**关于 $g$ 与固定尺度**：归一化后 $\text{RMS}(\hat q)=1\Rightarrow\|\hat q\|=\sqrt{d_{\text{head}}}$；乘 $g$ 后 $\|q_{\text{norm}}\|\approx\sqrt{d_{\text{head}}}\cdot\text{RMS}(g)$。关键：模长不再由**输入激活**决定，而由**参数 $g$** 决定。
- $g\equiv1$ 时 $\|q\|=\|k\|=\sqrt{d_{\text{head}}}$，logit $\le d_{\text{head}}$，经 $1/\sqrt{d_{\text{head}}}$ 缩放后量级 $O(\sqrt{d_{\text{head}}})$。
- $g$ 是**权重不是激活**：原不稳定是某些 token 激活在 forward 里被 attention 正反馈快速放大；$g$ 是所有 token/timestep 共享、靠梯度缓慢更新、受 weight decay 约束的慢变参数。每次 forward 激活模长都被 RMS 重置回 1，回路被掐断——attention 每步只能放大方向不能放大模长，模长闸门交给慢变参数。

所以「固定尺度」= 把模长从「随输入失控」改成「由慢变参数 $g$ 设定」，量级稳在 $\sqrt{d_{\text{head}}}\cdot\text{RMS}(g)$ 附近。

**为什么不用 LayerNorm**（论文未明确对照，以下为推断）：
1. LayerNorm 也能压模长（除以 std），但它额外的「减均值（centering）」对 Q/K 多余——注意力打分只关心方向与模长，沿特征维减「向量自身均值」几何上随意、抹掉一个自由度却换不来额外稳定性（稳定性来自「除以尺度」，RMSNorm 已具备）。
2. RMSNorm 更省（不算均值/bias，少一遍 reduction），对每 head 每 token 都做的 QK-Norm 不可忽略。

注意：QK-Norm 的归一化维度是 `head_dim`（每 head 的 Q/K 各自归一化），不是整个 hidden dim，所以能直接控制逐 head 的 $QK^\top$。

### Q7: 这篇文章除了上面几点，还有什么贡献？

1. **RF 时间步采样器族（3.1 节）**：不止 logit-normal（Eq. 19），还有 Mode Sampling with Heavy Tails（Eq. 20，端点密度严格为正，$s=0$ 退化 uniform）和 CosMap（Eq. 21–22，让 log-SNR 匹配 cosine）。
2. **统一加权损失框架（第 2 节）**：把 RF/EDM/Cosine/LDM-Linear/$\epsilon$-/$v$-prediction 纳入统一形式 $\mathcal L_w=-\frac12\mathbb E[w_t\lambda_t'\|\epsilon_\Theta-\epsilon\|^2]$，RF 对应 $w_t^{RF}=\frac t{1-t}$，并说明「改时间步采样 ⇔ 改损失加权」（Eq. 18）。
3. **16 通道 autoencoder（5.2.1 节，Table 3）**：增加 latent 通道数显著提升重建（$d=4\to8\to16$，FID 2.41→1.56→1.06），$d=16$ scaling 更好。区别于 SDXL 的 4 通道。
4. **合成 caption（5.2.2 节，Table 4）**：CogVLM 生成 + 50%合成/50%原始混合（纯合成会遗忘 VLM 知识库外概念），GenEval 43.27→49.78。
5. **可预测 scaling law（5.3.3 节，Figure 8）**：验证损失随规模/步数平滑下降无饱和（到 8B / $5\times10^{22}$ FLOPs）；**验证损失强相关于下游指标**（GenEval/人类偏好/T2I-CompBench）→ 主张验证损失作为通用性能度量；还做了视频初步 scaling。
6. **大模型采样更省步（Table 6）**：模型越大越少步达峰值，用 path length（$\sum\|v_\theta\cdot dt\|$，越大越直）解释——大模型更好拟合 RF 直线路径。
7. **多文本编码器 + T5 可丢弃（Figure 9）**：3 编码器各 46.3% dropout，推理可任意取子集；丢 T5 换显存，美学几乎无损、提示遵循略降，只有复杂排版/长文本明显需要 T5。
8. **混合宽高比位置编码（5.3.2 节）**：扩展+插值位置网格再频率嵌入+center-crop，配 bucketed sampling 支持灵活长宽比。
9. **数据预处理与安全缓解（5.3.1 节）**：NSFW/美学/去重过滤，latent 与文本 embedding 预计算缓存。
10. **DPO 对齐（Appendix C）**：1024² 训练后用 DPO，GenEval 最终 0.74，超 DALL·E 3 的 0.67。

贡献层次：方法层（统一损失框架 + RF 采样器族 + MM-DiT + QK-Norm + 分辨率 shift）→ 工程/数据层（16ch AE、合成 caption、多编码器、过滤、预计算、DPO）→ 科学层（可预测 scaling + 验证损失=性能代理 + 大模型更省步）。

## 我的收获 / 结论
- MM-DiT 的「双流独立权重 + joint attention 双向流」已成为 SD3/FLUX 等的通用范式，灵魂就在拼接 Q/K/V 后做一次联合注意力。
- 这篇在 diffusion 调参上的核心可借鉴点：**时间步采样要往中间倾斜（logit-normal $m=0,s=1$）**，比纠结选哪种 flow 更普适；本质是「让时间步分布对齐感知上 relevant 的噪声尺度」。
- 高分辨率的 resolution-dependent shift（$\alpha=3.0$）是上述思想在跨分辨率的延伸，单参数、等价 log-SNR 常数平移，实用。
- QK-RMSNorm 是稳训练的关键工程 trick：稳定性来自「除以尺度」，RMSNorm 比 LayerNorm 省且 centering 对 Q/K 多余。
- 「验证损失作为通用性能代理 + 可预测 scaling」是这篇能撑起「Scaling」标题的科学贡献，和那些工程决策（16ch AE、可丢 T5）一样重要。

## 关联工作 / 后续可读（可选）
- Rectified Flow / Flow Matching：Liu et al. 2022；Lipman et al. 2023；Albergo & Vanden-Eijnden 2022。
- DiT（Peebles & Xie 2023）、UViT（Hoogeboom et al. 2023）、PixArt-α（CrossDiT，Chen et al. 2023）。
- EDM（Karras et al. 2022）、统一加权损失视角（Kingma & Gao 2023）。
- RMSNorm（Zhang & Sennrich 2019）、QK-Norm 稳定性（Dehghani et al. 2023；Wortsman et al. 2023）。
- 合成 caption（DALL·E 3 / Betker et al. 2023）、CogVLM（Wang et al. 2023）。
- 后续：FLUX、SD3.5（沿用双流 MM-DiT，推断）。

## 关键摘抄（可选）
- "equivalent to having two independent transformers for each modality, but joining the sequences of the two modalities for the attention operation."
- 时间步加权等价：$w_t^\pi=\frac{t}{1-t}\pi(t)$（Eq. 18）。
- Resolution shift：$t_m=\frac{\sqrt{m/n}\,t_n}{1+(\sqrt{m/n}-1)t_n}$（Eq. 23），$\lambda_{t_m}=\lambda_{t_n}-\log\frac mn$（Eq. 25）。
- 全局排名第 1：`rf/lognorm(0.00, 1.00)`（Table 1，rank 1.54）。
- 规模：最大 8B（$d=38$），约 $5\times10^{22}$ 训练 FLOPs，scaling 无饱和迹象。
