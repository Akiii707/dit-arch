# VICreg-DiT: 视觉辅助训练传导至文生图的表征对齐方法

将 VICreg 视觉辅助训练的 VLM (Qwen3.5) 接入 OmniGen2 DiT 文生图管线，验证视觉自监督训练到文本表征的传导效应。

## 核心发现

- **VICreg 传导有效**：视觉自监督训练改善 VLM 文本表征，CLIP Score 提升 15-16%
- **VICreg 辅助 > 纯 SFT**：VICreg 辅助训练的表征传导效果优于纯 SFT 指令微调
- **对齐饱和是假象**：余弦相似度 ~0.96 不代表表征等价，信息量差异才是本质
- **LoRA 微调放大收益**：DiT 对嵌入分布的微调适应性进一步释放对齐潜力

## 方法架构

```
[文本输入] → [E2 VLM (Qwen3.5 + VICreg)] → [AlignMLP] → [DiT (LoRA)] → [VAE] → [输出图像]
```

- **E2 VLM**: Qwen3.5 架构 (hidden=2048)，视觉编码器经 VICreg 掩码潜空间辅助训练
- **AlignMLP**: 3层MLP (2048→4096→4096→2048) + scale，MSE+0.1cos 损失，fp32
- **DiT LoRA**: r=8, α=8, target=to_q/k/v/out.0，2500步微调

## 实验组

| 组 | VLM | AlignMLP | DiT | 说明 |
|---|---|---|---|---|
| Original | Qwen2.5-VL | 无 | 原版 | OmniGen2 原版基线 |
| Base (B0) | Qwen3.5-Base | align_mlp_base | 原版 | 换VLM+MLP对齐，无VICreg |
| E0_sft | Qwen3.5+VICreg+SFT | align_mlp_e0sft | 原版 | VICreg+SFT混合 |
| E2+LoRA (C1) | Qwen3.5+VICreg | align_mlp_real | LoRA微调 | 完整方法 |

## 目录结构

```
src/          # 核心代码
  t2i_orig.py       # 原版OmniGen2批量推理
  t2i_vlm_ft.py     # E2+LoRA推理
  t2i_vlm_b0.py     # Base/E0_sft对照推理
  merge_lora.py     # LoRA权重合并
  train_align.py    # AlignMLP训练(E2)
  train_align_base.py   # AlignMLP训练(Base)
  train_align_e0sft.py  # AlignMLP训练(E0_sft)
  clip_score_eval.py    # CLIP Score评测
  geneval_clip_eval.py  # GenEval分类别CLIP评测
  enhance_geneval_prompts.py  # Prompt增强脚本

scripts/      # 启动脚本
configs/      # Job配置
data/         # Prompt文件 (GenEval 553条 + 增强版)
docs/         # 论文初稿
results/      # CLIP Score结果
```

## 评测

### CLIP Score (20 prompt 快速评测)

| 组 | 长 Prompt | 短 Prompt |
|---|---|---|
| E2+LoRA | **0.2270** | **0.2145** |
| E0_sft | 0.2074 | 0.1891 |
| Base | 0.1976 | 0.1846 |

### GenEval Benchmark (553 prompt, 6类)

- single_object (80) / two_object (99) / counting (80) / colors (94) / position (100) / color_attr (100)
- 4模型 × 2 prompt版本 = 8组 × 553张 = 4424张图

## 依赖

- PyTorch 2.x + CUDA
- Transformers 5.14.1
- Diffusers 0.39.0
- PEFT 0.19.1
- CLIP ViT-L/14

## 引用

```bibtex
@misc{vicreg_dit_2026,
  title={VICreg视觉辅助训练传导至文生图的多模态表征对齐方法},
  author={Akiii707},
  year={2026},
  url={https://github.com/Akiii707/dit-arch}
}
```
