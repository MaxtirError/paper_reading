---
title: "GameFactory: Creating New Games with Generative Interactive Videos"
authors: "Jiwen Yu, Yiran Qin et al."
arxiv: "2501.08325"
url: "https://arxiv.org/abs/2501.08325"
year: 2025
date_read: "2026-06-26"
venue: "ICCV 2025"
category: "video-generation"
tags: [video-generation, game-generation, action-control, diffusion-models, world-model]
rating: 3
code: "https://yujiwen.github.io/gamefactory/"
status: read
---

# GameFactory: Creating New Games with Generative Interactive Videos

> 一句话总结（TL;DR）：利用预训练视频扩散模型的开放域生成先验，借助自采的无人类偏置 Minecraft 动作标注数据集 GF-Minecraft，通过 domain adapter + 多阶段解耦训练，把“动作控制”与“游戏风格”解耦，实现场景可泛化的动作可控游戏视频生成。

## 我的问答记录（QA）

### Q1: 这篇文章有训练模型吗？还是只 release 了一个 dataset？
不是纯数据集论文，有实际的模型训练流程。论文先用自采的 GF-Minecraft 动作标注数据集（Sec. 4.1），训练 action control module（Sec. 4.2），并用 LoRA 适配游戏风格（Sec. 5）。实验部分说明这些训练是基于一个**内部的 1B 规模 transformer-based text-to-video 扩散模型**进行的（Sec. 6.1）。核心贡献是“先学游戏风格、再学动作控制”的多阶段训练框架 GameFactory，数据集只是其中一部分。

### Q2: 这个数据集里面有包括鼠标的操作吗？鼠标的操作是怎么被注入的呢？
**有鼠标操作。** 动作空间表 Table 7 里鼠标被拆成两个连续信号：mouse movement (yaw) → vertical perspective movement（Interface4），mouse movement (pitch) → horizontal perspective movement（Interface5）。鼠标用指针相对游戏区域中心的偏移量表示，每帧取相对第一帧的累计绝对偏移作为输入（附录 A.3）。数据集刻意做成无人类偏置：键鼠都拆成原子动作并均衡分布。

**注入方式（Sec. 4.2，连续动作走 concat 路线）：**
1. 滑动窗口分组（窗口 w=3）解决 VAE 时间压缩率 r=4 带来的动作数 rn 与特征帧数 n+1 的粒度不匹配，并捕捉延迟效应，得到 $M_{group}\in\mathbb{R}^{(n+1)\times rw\times d_1}$；
2. reshape → 沿 token 维度 repeat → 沿**通道维度**与特征 F 拼接得到 $F_{fused}$；
3. 再过一层 MLP + 一层 temporal self-attention。

消融（Table 2）证明：连续鼠标用 concatenation 比 cross-attention 好（cross-attn 的相似度计算会削弱信号幅值的影响），而离散键盘用 cross-attention 更好。即：**键盘 = cross-attention 注入，鼠标 = 分组后通道拼接 + MLP + 时序自注意力**。

### Q3: 这篇文章有 camera 注入吗？还是纯 action-based 的？backbone 用的什么？
**纯 action-based（键盘 + 鼠标），没有显式 camera pose 注入。** 它不像 MotionCtrl / CameraCtrl / Direct-a-Video 那样把相机外参作为条件——这些只在 Related Work 被当作对比对象提及。相机/视角运动是通过**鼠标动作（yaw/pitch）**间接驱动的，注入方式仍是上面的 concat 路线。论文里的 "Cam" 是**评测指标**（用 GLOMAP 从生成视频反推相机位姿再比欧氏距离），不是条件输入。

**Backbone：** transformer-based latent video diffusion model（DiT 路线，Sec. 3）。实验实例是一个**内部 1B 规模 text-to-video 扩散模型，由更大的预训练视频扩散模型蒸馏而来**。关键超参：VAE 时间压缩率 r=4、分辨率 360×640、DDIM 50 步采样。

### Q4: 它是和什么 baseline 对比的？
**几乎没有跟其他方法做统一跑分**，作者明说各方法数据来源/分辨率/控制粒度不同，难建统一基准（Sec. 6.1）。对比分几类：
- **Table 1（定性特性表，非跑分）**：与 DIAMOND、GameNGen、GameGenX、Oasis、Matrix、Genie 2 并排比特性（是否可测试模型 / 是否公开数据集 / 动作空间 / 是否场景泛化）。
- **Table 2（消融）**：cross-attention vs concatenation 注入机制。
- **Table 3（消融）**：Multi-Phase vs One-Phase 训练，以 in-domain 为参照。
- **Table 4 & 5（数据集对比）**：自家 GF-Minecraft vs VPT，证明 VPT 有严重人类偏置。
- **Table 6（消融）**：loss 算所有帧 vs 仅预测帧。

即主要 baseline 是它自己的变体 + VPT 数据集，对同类方法只有特性表层面的定性对照。

### Q5: dataset 是自己收集的对吧？
对，自采，叫 **GF-Minecraft**（GF = GameFactory）。用 MineDojo 执行预定义动作序列采集，共 **70 小时**视频；附录 A.1 给出更细的数字：**2,000 个片段、每个 2,000 帧**。不用现成 VPT 是因为后者来自真人游玩、带人类偏置且无文字描述。消除偏置：键鼠拆原子动作 + 均匀采样频率/时长 + 随机组合 + 随机化持续帧数。场景多样性：3 生物群系 / 3 天气 / 6 时间段，并用 MiniCPM-V 自动生成文字标注。Table 1 里 Available Dataset 只有 GameGenX 和它打勾。

### Q6: 这篇的叙事逻辑偏向实验还是数据集？contribution 类型？
本质是一篇 **method/framework 论文**，不是纯实验驱动也不是纯数据集驱动。三条贡献里第 1（框架）和第 3（domain adapter + 多阶段解耦训练）都是方法，数据集排第 2 且大量细节被放进附录 A。主线叙事是“预训练先验 → 直接微调会被游戏风格污染 → 用 domain adapter + 多阶段训练解耦风格与动作控制”，是典型方法论叙事。实验几乎全是消融（验证设计选择），没有刷榜。贡献类型 = **新框架 / 新训练范式 > 新数据集 > 实验对比**。作者还把它升格定位为 “Generalizable World Model”（附录 C）。

## 我的收获 / 结论
- 我觉得这套叙事**不太行**：同类型工作（DIAMOND / Oasis / Matrix / Genie 2 / GameGenX 等）很多，却没有做实质性的数值对比，只有一张定性特性表，说不太过去。
- 作者用“统一基准难建”来解释不对比，但这在如此拥挤的赛道里说服力有限。
- Anyway，这篇中了 **ICCV 2025**。

## 关联工作 / 后续可读（可选）
- VPT [3]（被当作有人类偏置的数据集 baseline）
- Matrix [14]、Genie 2 [12]、Oasis [11]、GameGenX [7]、DIAMOND [2]、GameNGen [42]（Table 1 横向对比对象）
- Diffusion Forcing [8]（自回归长视频生成的灵感来源）
- MotionCtrl / CameraCtrl / Direct-a-Video（相机控制类对比，本文未采用）

## 关键摘抄（可选）
- "we propose a multi-phase training strategy with a domain adapter that decouples game style learning from action control."（Abstract）
- 鼠标融合：$M_{group}\in\mathbb{R}^{(n+1)\times rw\times d_1}$ → repeat → 沿通道与 $F$ 拼接 → MLP + temporal self-attention（Sec. 4.2）。
- "establishing a unified benchmark for comparison is challenging."（Sec. 6.1，作者对不做统一对比的解释）
