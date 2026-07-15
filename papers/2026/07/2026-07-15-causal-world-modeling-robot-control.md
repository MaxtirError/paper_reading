---
title: "Causal World Modeling for Robot Control"
authors: "Lin Li et al."
arxiv: "2601.21998"
url: "https://arxiv.org/abs/2601.21998"
year: 2026
date_read: "2026-07-15"
venue: "arXiv preprint"
category: "robot-learning"
tags: [world-model, action-control, robot-manipulation, diffusion-models, transformer]
rating: 5
code: "https://github.com/robbyant/lingbot-va"
status: read
---

# Causal World Modeling for Robot Control

> TL;DR：LingBot-VA 用因果自回归的视频-动作世界模型统一未来视觉预测与逆动力学，通过真实观察闭环、KV cache、部分视频去噪和异步执行支持长时机器人操作；但论文描述的 separated MoT 与 FDM 训练细节尚未公开。

## 我的问答记录（QA）

### Q1: 我看到了poster上有一张图，里面写着video world model, inverse dynamics 和robō t execution，这几个分别是什么意思

三者构成“先想象未来，再反推动作，最后执行并接收真实反馈”的闭环（图 1，§3.1–3.2，公式 6、8、9）：

- **Video world model**：根据视觉与动作历史预测接下来的视觉 latent，即 future imagination：
  $$z_{t+1:t+K}\sim p_\theta(\cdot\mid z_{\le t},a_{<t}).$$
- **Inverse dynamics**：给定当前状态和期望/预测的未来状态，反推出实现该视觉变化所需的动作：
  $$a_{t:t+K-1}\sim g_\psi(\cdot\mid \hat z_{t+1:t+K},z_{\le t},a_{<t}).$$
- **Robot execution**：机器人执行动作，摄像头获得真实新观察，再用它更新上下文并规划下一步。

这只是概念分解；模型架构中视频 token 与动作 token 被放入统一的自回归序列联合建模。

### Q2: 我有一个问题，为什么要用O_{T+1}去修正之前的想象，既然有摄像头得到的真实的新观察，为什么不直接用真实的观察去做逆动力学这个操作呢？

决定 $A_t$ 时，真实 $O_{t+1}$ 尚未发生，因为它是执行 $A_t$ 后才得到的结果。在线推理只能先用预测的 $\hat O_{t+1}$ 作为目标，通过 $(O_t,\hat O_{t+1})$ 推断 $A_t$。执行后得到真实 $O_{t+1}$，其作用不是重算已经执行的 $A_t$，而是重新锚定现实并用于预测 $\hat O_{t+2}$、生成 $A_{t+1}$。

训练时完整示范轨迹已经存在，所以逆动力学可以使用真实的 $(O_t,O_{t+1})$；部署时不能提前获得真实未来。

### Q3: 这个所谓的修正应该就是直接替换对吧

同步闭环中可以理解为直接替换：真实观察到达后，不再把旧预测当作当前现实，而是编码 $z_{t+1}=E(O_{t+1})$ 并从它继续预测。

异步模式更准确地说是：删除过时预测，以最近真实观察为新起点，结合正在执行的动作通过 FDM 重建新的预测后缀。它不是 $\hat O+\Delta O$ 式像素修补，而是更新上下文并重新预测。

### Q4: 翻译一下论文里的这段描述，说明一下这个操作：To address this, we initialize the action network weights by interpolating the pretrained video weights according to the action dimension, then apply a scaling factor α = sqrt(dv/da) to preserve output variance...

翻译：作者按动作网络维度对预训练视频网络权重进行插值，用其初始化动作网络，再乘

$$\alpha=\sqrt{d_v/d_a}$$

保持输出方差，使动作 token 在训练开始时具有与视频 token 相近的输出分布，从而稳定早期优化并加速收敛。

论文设 $d_v=3072,d_a=768$，所以 $\alpha=2$。若宽度缩小四倍，线性变换的累加项减少；权重乘 2 会令权重方差扩大四倍，从而近似补偿维度变化：

$$d_a\alpha^2=d_v.$$

**我认为这个初始化操作很有启发。** 但论文没有给出具体插值算法，且当前公开 shared-backbone 代码没有这条独立动作分支，因此精确实现仍未知。

### Q5: 一个独立问题：MoT架构是什么样的？意思是两个独立transformer，然后中间cross attention是吗？

论文描述的 separated MoT 是同深度、不同宽度的两条分支：视频分支 $d_v=3072$，动作分支 $d_a=768$。二者保留模态专属参数，在每层 attention 中交换信息，而不是两个完整 Transformer 最后才接一次 cross-attention。

联合 attention 的矩阵可分为：

$$
\begin{bmatrix}
A_{vv}&A_{va}\\
A_{av}&A_{aa}
\end{bmatrix},
$$

一次计算同时包含视频 self-attention、动作 self-attention及双向跨模态交互；之后两条分支各自做输出投影、残差、归一化与 FFN，一般无需紧接着再重复独立 self-attention。T5 文本指令则通过标准 cross-attention 注入。

### Q6: 做了这个joint attention之后还会做各自模块的self attention吗？还是就不做了，还有，为什么不直接在toQ to K, to V改projection的维度还要加一个Pup

Joint attention 已包含两侧 self-attention，因此通常不再额外重复一次。

从线性代数上，直接令动作 QKV 将 $768\to3072$ 完全可行，显式 $P_{up}$ 不是数学必需。论文描述中的 $P_{up}/P_{down}$ 更像一种参数化选择：分离模态对齐与 QKV 投影、保持 768 维动作 bottleneck，并保留低维残差路径。但未发布的 separated MoT 代码尚不可核实该细节。

### Q7: 我吧代码下载到code文件夹里面了，你能细说一下这部分的实现吗？最好结合代码

当前公开代码与论文描述不同：README 明确标注发布的是 **shared-backbone** 版本，separated MoT 尚未发布。

公开模型把视频 patch 和动作分别映射到同一个 3072 维空间：

```python
self.patch_embedding_mlp = nn.Linear(48 * 1 * 2 * 2, 3072)
self.action_embedder = nn.Linear(30, 3072)
```

训练时拼接 noisy video、clean video、noisy action、clean action，统一通过同一套 30 层 `WanTransformerBlock` 和共享 QKV。每个 block 是：统一 self-attention、对 T5 文本的 cross-attention、共享 FFN。训练端用 FlexAttention mask 建立逻辑因果关系；推理端先生成视频并写入 KV cache，再用同一主干生成动作。因此公开版本不存在独立 768 维动作 Transformer、显式 $P_{up}/P_{down}$ 或权重插值初始化。

### Q8: 有说不release MoT版本的原因吗？

没有。作者只说正在准备并会在 ready 后发布，没有说明技术、许可、算力或其他原因，也没有时间表。GitHub Issue #4 明确说 MoT 是独立架构，不是 shared backbone 的概念解释。

作者在 Issue #10 还说明论文 RoboTwin 报告结果来自已发布的 shared-backbone checkpoint。因此目前无法验证 separated MoT 的具体实现与增益，构成明显的可复现性缺口。

### Q9: Teacher Forcing Attention Mask 里面的z1(s)是什么意思，这部分对应的代码在哪里？

$z_1^{(s)}$ 是干净视频 latent $z_1$ 在 flow time $s$ 上的带噪插值：

$$z_1^{(s)}=(1-s)\epsilon+sz_1.$$

它是当前位置的去噪输入，不是第 $s$ 帧。训练代码在 `train.py::_add_noise()` 中采样 timestep 和噪声，由 `FlowMatchScheduler.add_noise()` 构造 `noisy_latents`。代码使用反向参数 $\sigma$：

$$z^{(\sigma)}=(1-\sigma)z+\sigma\epsilon,\qquad \sigma=1-s.$$

`noisy_latents` 对应 $z^{(s)}$，`latent` 对应 clean teacher-forcing history，`targets` 是 flow velocity 监督。

### Q10: Emm，我其实对teacher forcing这个概念一直有一个不太理解的地方，是这样的，就是按照道理来讲，我正常做bidrectional的时候，就是一个（clean_latent + noise_latent）的full attention，关键是，这个时候clean比如说是z0~z2,那么noise就一定是z3（s）~.....，不会有 z0(s)，我的问题就是，为什么teacher forcing 的时候z0和z0(s)会一起出现，含义是什么，是只有training这么做，还是train,infer都这么做

同一位置的 clean $z_t$ 与 noisy $z_t^{(s)}$ 同时放入训练张量，不是让它们互相 attention，而是为了一个 forward 并行训练所有自回归位置：

$$z_1^{(s)}\leftarrow z_0,$$
$$z_2^{(s)}\leftarrow z_0,z_1,$$
$$z_3^{(s)}\leftarrow z_0,z_1,z_2.$$

其中 $z_t^{(s)}$ 是当前位置的预测输入，clean $z_t$ 只作为后续位置的真实历史。Mask 禁止 $z_t^{(s)}$ 读取同位置 clean $z_t$，避免标签泄漏。

这主要是训练技巧。推理时未来 clean token 不存在，只维护已经获得的真实/预测历史和当前 noisy chunk。

### Q11: 那么就有一个新的疑问了，这么做对于train和test的计算behave是一致的吗？从attention的公式出发分析一下

对 query $q_t$，训练用 mask 将允许集合记为 $\mathcal A_t$：

$$o_t^{train}=\sum_{j\in\mathcal A_t}\frac{\exp(q_t^Tk_j/\sqrt d)}{\sum_{l\in\mathcal A_t}\exp(q_t^Tk_l/\sqrt d)}v_j.$$

推理时不合法 token 根本不存在，KV cache 中也只有同一集合 $\mathcal A_t$，所以在输入、位置编码和参数相同的条件下，masked full-sequence attention 与增量 KV-cache attention 数学等价。

但输入分布不完全一致：训练主要使用真实 clean history，推理历史可能包含预测 token。故计算图和因果范围一致，不代表历史分布及误差累积一致。

### Q12: ok，然后请你解释一下noise augmentation的机制和kv cache的算法

需区分普通 flow target 加噪和 **Noisy History Augmentation**。后者以 0.5 概率将历史视频 latent 变成：

$$\tilde z_{\le t}=(1-s_{aug})\epsilon+s_{aug}z_{\le t},\qquad s_{aug}\in[0.5,1].$$

其论文明确目的，是让逆动力学模型能从部分去噪的视频表示提取动作信息，从而视频侧可以提前停止去噪，降低 inference 延迟。它不等价于纠正错误未来或解决长期幻觉漂移。

KV cache 在每个 Transformer 层保存历史 token 的 $K=XW_K$ 和 $V=XW_V$。新一步只计算当前 token 的 QKV，再让当前 query 读取缓存：

$$o_t=\operatorname{softmax}\left(\frac{q_t[K_{cache};k_t]^T}{\sqrt d}\right)[V_{cache};v_t].$$

公开代码区分三类更新：`update_cache=0` 为当前去噪步临时使用，算完撤回；`1` 持久保存模型预测并标记 `is_pred=True`；`2` 保存真实反馈并标记为非预测。真实观察到达后，`clear_pred_cache()` 先使预测槽位失效，再将真实 observation 和已执行动作重新通过所有层，生成新的 K/V 写入缓存。

### Q13: Noisy History Augmentation 让动作模型“看得懂不完美的想象” 这个东西我怎么觉得是你的幻觉呢？我看原文里面Noisee History Augmentation就是为了加速inference的吧

这个质疑是正确的。准确说法是：**Noisy History Augmentation 让逆动力学适应部分去噪的视频 latent，使视频生成提前停止，从而加速 inference。**

“不完美的想象”容易把部分去噪噪声、世界模型语义预测错误和长期幻觉漂移混为一谈。论文没有证明该机制可以修正错误未来；长期漂移主要由真实观察闭环和 FDM-grounded async 处理。

### Q14: 然后接下来继续细说一下Asynchronous Inference and Execution首先这个stale prediction是什么意思，还有就是他这个算法里面我能get的是他temporally使用的是predict的frame然后直接当成clean 的history去rollout，但是当真机observation得到了之后，如何替换kv cache这个过程我没有get，能详细解释一下吗？

**Stale prediction** 是基于较旧 observation 生成、没有利用后来到达的最新真实反馈的过时预测。Naive async 会继续沿用它，视频模型容易延续自己的幻觉轨迹。

FDM-grounded async 维护已确认历史 $C$ 和临时分支 $C_{tmp}$。机器人执行当前动作时，模型用最近真实 observation 与当前动作建立临时预测视觉，再规划下一动作。真实 observation 到达后：

1. 将 `is_pred=True` 的缓存槽位置为无效；
2. 编码真实 observation；
3. 将真实 observation 和已执行动作重新跑过所有 Transformer 层；
4. 以 confirmed K/V 重建缓存。

因此不是原位改写 $KV(\hat z)$，而是回滚预测后缀并重算 $KV(z^{real})$。当前公开评测循环实现了缓存回滚/重写，但没有完整呈现论文图 4 中机器人执行与 GPU 推理真正并行的双线程调度。

### Q15: 所以我的理解是，对于模型的prediction而言，假如我要rollout an+1，每次实际上执行的context是由Real(z0....zn-1an-1) + Predict(zn)得到的？会有一个chunk的predict fake visual hint?

是的，但还应加入当前正在执行的动作：

$$\underbrace{(z_{\le n-1}^{real},a_{<n}^{executed})}_{confirmed} + a_n^{executing}+\hat z_n^{FDM}.$$

为了在真实 $z_n$ 尚未到达时提前生成 $a_{n+1}$，context 中临时存在一个由最近真实观察和当前动作推演出的预测视觉 chunk，充当 visual hint。真实 $z_n$ 到达后，该预测 chunk 被回滚并由真实状态替换。

### Q16: 那这个 forward dynamics prediction loss:是什么意思

FDM loss 训练：给定当前真实状态和当前动作，预测动作执行后的视觉状态：

$$(z_t,a_t)\rightarrow z_{t+1}.$$

它与 inverse dynamics 的 $(z_t,z_{t+1})\rightarrow a_t$ 方向相反。公式 (13) 仍是对目标视频 latent 做 flow-matching velocity MSE，但条件显式包含当前动作，用于异步推理时从最新真实观察重建 grounded prediction。

### Q17: 那问题来了，这个forward dynamics的attention mask怎么给呢？看起来forward dynamic和dynamic两个对应的attention mask不同啊

普通 dynamics 预测目标视觉时不能看到对应的尚未生成动作；FDM 中该动作已经在执行，目标视觉必须能读取它，所以条件集合确实不同。

**论文及当前公开代码没有给出 FDM loss 的具体训练实现。个人推测：post-training 使用额外的 FDM loss，并配套不同于普通 dynamics 的 FDM attention mask，使目标视觉能够读取当前执行动作。** 这只是推测，不能写成已确认实现。

### Q18: 关于Implementation，他diffusion sample只分别用了3步和10步对吗？那它有蒸馏吗？

论文 §4.2 报告视频 latent 用 3 个 Euler steps、积分到约 $s=0.6$；动作 latent 用 10 个 Euler steps、完整积分到 $s=1.0$。Video CFG 为 5.0，Action CFG 为 1.0。

论文未报告任何 teacher-student、consistency 或 progressive distillation。视频侧只走 3 步的依据是动作解码只需部分去噪 latent，而不是把完整高质量视频生成器蒸馏成 3 步模型。§3.3/Algorithm 1 写 $s=0.5$，§4.2 实验配置写 $s=0.6$，实际参数应以 §4.2 为准。

### Q19: 虽然没有release MoT，但是论文这里说 The action stream shares the same depth but uses a reduced hidden dimension da = 768 (4× smaller), resulting in approximately 350M additional parameters and a total model size of 5.3B parameters. 为什么减少dimension反而模型变大了？

动作分支不是替换视频网络，而是在约 5B 视频主干之外额外增加一条同样 30 层、宽度 768 的 Transformer。因此总参数是：

$$N_{total}=N_{video}+N_{action}+N_{projection}\approx5.0B+0.35B=5.3B.$$

Transformer 参数量近似随 $d^2$ 缩放，动作宽度为视频的四分之一，参数约为完整视频分支的 $1/16$：$5B/16\approx312.5M$，再加投影等得到约 350M。减少宽度只是控制新增成本，总参数仍会增加。

### Q20: During training, chunk size K is randomly sampled from [1, 4] 这个chunk对应的是多少帧？不会是wan的80帧吧

不是 Wan 常见的 80/81 帧 clip。$K$ 是一个 AR chunk 中并行生成的 **VAE latent 时间位置数**。训练随机取 1–4，mask 用 $\lfloor f/K\rfloor$ 将连续 latent positions 划为一个逻辑 chunk；chunk 内双向 attention，chunk 间保持因果。

Wan causal VAE 时间压缩约为 4，因此 $K=1$ 到 4 粗略对应约 4–16 个原始 RGB 时间帧，但边界处不应机械换算。论文概念上每个稀疏视频位置关联 $\tau=4$ 个高频动作；工程配置会依控制频率变化，例如 LIBERO 为 4 latent frames、每 frame 4 actions，RoboTwin 当前配置为 2 latent frames、每 frame 16 actions。

### Q21: 为什么还要经历一个pre-train+post-train的过程

严格说有三层：Wan2.2 视频预训练、LingBot-VA 大规模 video-action pre-training、目标平台 post-training。

Wan 提供通用视频先验；LingBot-VA pre-training 用约 16K 小时、多 embodiment 机器人数据学习通用视觉动力学和视觉-动作关系；post-training 再适配具体机器人的几何、相机、动作坐标、频率和任务。这样可一次大规模训练 base model，再低成本适配多个下游平台。

### Q22: 总结一些实验中的每个表格，并简单告诉我数据集和指标的含义

- **RoboTwin 2.0**：50 个双臂仿真任务。Easy 固定初始配置，Hard 随机物体姿态与场景；Horizon 表示主要操作阶段数。LingBot-VA 平均 SR 为 92.93%/91.55%，且长 Horizon 相对优势更大。
- **LIBERO**：Spatial、Object、Goal、Long 四个 suite，每套 10 个任务。LingBot-VA 平均 SR 98.5%，LIBERO-Long 98.5%。
- **SR**：完整成功 trial 比例；**PS**：按中间步骤计分的任务进度，首次成功 1、重试成功 0.5、失败 0。
- **表 3 消融**：FDM-grounded Async 90.4%，Naive Async 74.3%，Horizon 3 为 85.6% 对 32.9%；WAN 初始化 80.6%，低于 LingBot-VA 预训练的 92.9%。但表中缺少 caption 宣称的 AR vs bidirectional 和明确同步速度数据。
- **真实任务附表**：Make Breakfast 97.0 PS/75 SR；Pick Screws 82.5/70；Unpack Delivery 84.5/65；Insert Tubes 85.8/40；Fold Pants 76.7/70。Fold Clothes 是反例：LingBot-VA SR 35 高于 $\pi_{0.5}$ 的 30，但 PS 48.8 低于 62.9，与正文“所有任务、两个指标均领先”的表述冲突。

### Q23: Ok，首先是这样的，我还是没get post-train，作者说能在robotwin上fine-tune，为什么不直接在pre-training阶段加入robotwin上的数据？post-train这个步骤是这个领域的标准操作吗？

技术上可以把 RoboTwin 混入 pre-training，但单独 post-training 能保留可复用的通用 base model，以较低成本适配目标 embodiment，并更清楚地评估迁移与 sample efficiency。大规模通用预训练加目标平台/任务 supervised fine-tuning 是 VLA 和机器人基础模型中的常见流程，也常称 downstream adaptation、embodiment adaptation 或 SFT。

需要注意 RoboTwin 主实验并非总共只用 50 条数据：50 个任务各有 50 条 clean demos 加 500 条 heavily randomized demos，总计 27,500 条，并训练 50K steps。Post-training 主要是目标分布微调；论文还称加入 FDM loss，但其训练细节未公开。

## 我的收获 / 结论

- 我很认可“视频世界模型先想象未来、逆动力学再反推动作”的分解，以及动作分支通过视频权重插值并做方差补偿的初始化思路；后者尤其值得借鉴。
- Teacher forcing 的 clean/noisy 双份 token，本质是把多个不同前缀的自回归训练样本打包进一次 masked forward，而不是让同位置 noisy token 读取 clean label。
- Noisy History Augmentation 的准确作用是支持部分视频去噪、加速 inference，不能延伸为纠正错误未来或长期幻觉漂移。
- 异步推理依赖一个临时预测视觉 chunk 跨越真实反馈延迟；真实 observation 到达后，回滚 speculative KV 后缀并重建 confirmed cache。
- FDM loss 的训练细节不明。个人推测 post-training 有额外 FDM loss 和对应 FDM attention mask，但论文及公开代码均未提供实现证据。
- 当前公开代码是 shared-backbone，不是论文描述的 separated MoT；MoT 代码、权重插值细节和对应增益仍不可复现。
- 实验整体很强，但表 3 缺少若干 caption/正文声称的消融与速度数据，Fold Clothes 的 PS 也与“所有指标领先”的文字结论冲突。

## 关联工作 / 后续可读

- Wan2.2：视频生成主干与 causal VAE。
- Mixture-of-Transformers：论文 separated 双流架构的来源。
- Motus / UWM / UVA：统一视频-动作世界模型。
- $\pi_0$ / $\pi_{0.5}$：主要 VLA 对照方法。
- Seer：predictive inverse dynamics robot policy。

## 关键摘抄

> “The key insight is that action prediction does not require fully denoised video representations; instead, it can rely on robust semantic structures.”

> “Currently, we have only released the weights and code for the shared backbone. We are actively preparing the MoT version and will release it as soon as it’s ready.”（作者 GitHub Issue #4）

关键初始化：

$$\alpha=\sqrt{d_v/d_a}=\sqrt{3072/768}=2.$$

关键边界：论文称 post-training 额外加入 FDM loss，但其训练 attention mask 与数据构造未公开。
