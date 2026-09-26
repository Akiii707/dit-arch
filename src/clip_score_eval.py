"""CLIP Score 评测：对三方出图（batch20/batch20_b0/batch20_e0sft）计算图文匹配度。

指标：CLIP cosine similarity（image embedding vs text embedding）。
每组输出 per-image 分数 + 组均值，三方对比。
"""
import argparse, os, json
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


def load_prompts(path):
    """读 t2i_prompts_20.txt，返回 [(name, prompt), ...]"""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, prompt = line.split("|", 1)
            items.append((name, prompt))
    return items


@torch.no_grad()
def compute_clip_score(clip_model, processor, images, texts, device):
    """批量计算 CLIP score（cosine similarity）。"""
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)
    img_feats = outputs.image_embeds  # [N, 512]
    txt_feats = outputs.text_embeds   # [N, 512]
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    scores = (img_feats * txt_feats).sum(dim=-1)  # [N]
    return scores.cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/clip-vit-large-patch14")
    ap.add_argument("--prompts_file", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/v_launch/t2i_prompts_20.txt")
    ap.add_argument("--t2i_root", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i")
    ap.add_argument("--groups", nargs="+", default=["batch20", "batch20_b0", "batch20_e0sft"])
    ap.add_argument("--output_json", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i/clip_score_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP from {args.clip_path} ...", flush=True)
    clip_model = CLIPModel.from_pretrained(args.clip_path, torch_dtype=torch.float32).eval().to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    print(f"CLIP loaded on {device}", flush=True)

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    all_results = {}
    for group in args.groups:
        group_dir = os.path.join(args.t2i_root, group)
        if not os.path.isdir(group_dir):
            print(f"[SKIP] {group}: directory not found", flush=True)
            continue
        print(f"\n=== Evaluating {group} ===", flush=True)
        per_image = []
        images, texts, names = [], [], []
        for name, prompt in prompts:
            img_path = os.path.join(group_dir, f"{name}.png")
            if not os.path.exists(img_path):
                print(f"  [MISSING] {name}.png", flush=True)
                per_image.append({"name": name, "prompt": prompt, "score": None, "missing": True})
                continue
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            texts.append(prompt)
            names.append(name)

        if images:
            scores = compute_clip_score(clip_model, processor, images, texts, device)
            for i, (name, prompt) in enumerate(zip(names, texts)):
                s = scores[i].item()
                per_image.append({"name": name, "prompt": prompt, "score": round(s, 4)})
                print(f"  {name}: {s:.4f}", flush=True)

        valid_scores = [x["score"] for x in per_image if x.get("score") is not None]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        all_results[group] = {
            "num_images": len(valid_scores),
            "mean_clip_score": round(avg, 4),
            "per_image": per_image,
        }
        print(f"  >>> {group} mean CLIP score: {avg:.4f} ({len(valid_scores)} images)", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_json}", flush=True)
    print("\n=== Summary ===", flush=True)
    for group, data in all_results.items():
        print(f"  {group}: {data['mean_clip_score']:.4f} ({data['num_images']} images)", flush=True)


if __name__ == "__main__":
    main()
