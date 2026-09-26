"""把 checkpoint-2500 中的 LoRA 权重合并进基础权重，输出合并后的 transformer state dict。

LoRA 配置（train.py L341-346）：r=8, lora_alpha=8 => scaling=1.0
checkpoint 键格式：{module}.lora_A.default.weight / {module}.lora_B.default.weight
基础权重键：{module}.weight
合并公式：W' = W + lora_B @ lora_A * (alpha / r)
"""
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2/experiments/ft_lora_e2/checkpoint-2500/pytorch_model_fsdp.bin")
    parser.add_argument("--output", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2/experiments/ft_lora_e2/checkpoint-2500/transformer_merged.bin")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    args = parser.parse_args()
    scaling = args.alpha / args.rank

    print("loading full FSDP state dict (mmap)...", flush=True)
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    keys = list(sd.keys())
    lora_keys = [k for k in keys if ".lora_" in k]
    base_keys = [k for k in keys if ".lora_" not in k]
    print(f"total={len(keys)} base={len(base_keys)} lora={len(lora_keys)}", flush=True)

    merged = {}
    for k in base_keys:
        merged[k] = sd[k].clone()

    n_merged = 0
    a_keys = {k[:-len(".lora_A.default.weight")] for k in lora_keys if k.endswith("lora_A.default.weight")}
    for mod in sorted(a_keys):
        a = sd[f"{mod}.lora_A.default.weight"].float()
        b = sd[f"{mod}.lora_B.default.weight"].float()
        # peft 包装后基础权重存为 {mod}.base_layer.weight，去掉包装后是 {mod}.weight
        wkey = f"{mod}.weight"
        src_key = f"{mod}.weight"
        if wkey not in merged and f"{mod}.base_layer.weight" in merged:
            src_key = f"{mod}.base_layer.weight"
        assert src_key in merged, f"base weight missing: {wkey} / {mod}.base_layer.weight"
        w = merged[src_key].float()
        delta = (b @ a) * scaling
        assert delta.shape == w.shape, f"shape mismatch {mod}: {tuple(delta.shape)} vs {tuple(w.shape)}"
        merged[src_key] = (w + delta).to(sd[src_key].dtype)
        n_merged += 1
    print(f"merged {n_merged} lora pairs, scaling={scaling}", flush=True)

    # 抽样校验：与基础权重差异不能为零（说明确实学到了东西），也不能爆炸
    ck = next(iter(a_keys))
    ck_src = f"{ck}.base_layer.weight" if f"{ck}.base_layer.weight" in sd else f"{ck}.weight"
    dw = (sd[ck_src].float() - merged[ck_src].float()).abs()
    print(f"sample module {ck}: delta abs mean={dw.mean().item():.6f} max={dw.max().item():.6f}", flush=True)

    # 输出去掉 peft 包装的键名：{mod}.base_layer.weight -> {mod}.weight
    final = {}
    for k, v in merged.items():
        final[k.replace(".base_layer.weight", ".weight")] = v
    print(f"renamed base_layer keys: {len(merged) - 0} -> final {len(final)} keys", flush=True)

    torch.save(final, args.output)
    print(f"saved merged transformer state dict to {args.output}", flush=True)

if __name__ == "__main__":
    main()