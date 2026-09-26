"""B0 对照组推理：Qwen3.5-2B-Base + align_mlp_base + 原版 DiT（不加载微调权重）。

与 t2i_vlm_ft.py 的 C1 组唯一差异：文本编码器换 base、DiT 用原版权重。
prompt/seed/推理参数与 C1 完全一致，输出到 batch20_b0/ 供逐图对比。
"""
import argparse, os
import torch
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusers import AutoencoderKL


class AlignMLP(torch.nn.Module):
    def __init__(self, dim=2048, hidden=4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, dim),
        )
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.net(x) * self.scale


class E2VLMWrapper(torch.nn.Module):
    def __init__(self, model, align_mlp=None):
        super().__init__()
        self.model = model
        self.align_mlp = align_mlp

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states and self.align_mlp is not None:
            h = out.hidden_states[-1]
            mlp_dtype = next(self.align_mlp.parameters()).dtype
            h = self.align_mlp(h.to(mlp_dtype)).to(h.dtype)
            out.hidden_states = tuple(out.hidden_states[:-1]) + (h,)
        elif output_hidden_states:
            out.hidden_states = tuple(out.hidden_states)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2/pretrained_models")
    parser.add_argument("--vlm_path", default="/data/juicefs_sharing_data/public_data/11195663/model/Qwen3.5-2B-Base")
    parser.add_argument("--align_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/align_base/align_mlp_base.pt")
    parser.add_argument("--prompts_file", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/v_launch/t2i_prompts_20.txt")
    parser.add_argument("--output_dir", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i/batch20_b0")
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    args = parser.parse_args()
    root = os.path.abspath(args.model_path)
    dtype = torch.bfloat16

    print("loading scheduler...", flush=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    print("loading vae...", flush=True)
    vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    print("loading transformer (original, NOT finetuned)...", flush=True)
    transformer = OmniGen2Transformer2DModel.from_pretrained(root, subfolder="transformer", torch_dtype=dtype)

    print("loading Qwen3.5-2B-Base...", flush=True)
    mllm = E2VLMWrapper(
        Qwen3_5ForConditionalGeneration.from_pretrained(args.vlm_path, torch_dtype=dtype)
    ).eval()
    mlp = AlignMLP()
    mlp.load_state_dict(torch.load(args.align_path, map_location="cpu"))
    mllm.align_mlp = mlp.eval()
    print(f"loaded align_mlp from {args.align_path} (fp32)", flush=True)
    print("loading processor (tokenizer replaced by base)...", flush=True)
    processor = AutoProcessor.from_pretrained(os.path.join(root, "mllm_processor"))
    processor.tokenizer = AutoTokenizer.from_pretrained(args.vlm_path)

    pipeline = OmniGen2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor)
    accelerator = Accelerator(mixed_precision="bf16")
    pipeline = pipeline.to(accelerator.device)
    for p in mllm.parameters():
        p.requires_grad_(False)

    with open(args.prompts_file) as f:
        tasks = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    print(f"B0 batch: {len(tasks)} prompts", flush=True)
    for idx, line in enumerate(tasks):
        name, prompt = line.split("|", 1)
        out_path = os.path.join(args.output_dir, f"{name}.png")
        try:
            generator = torch.Generator(device=accelerator.device).manual_seed(0)
            with torch.no_grad():
                results = pipeline(prompt=prompt, input_images=None, width=1024, height=1024,
                                   num_inference_steps=args.num_inference_step, max_sequence_length=1024,
                                   text_guidance_scale=args.text_guidance_scale, image_guidance_scale=2.0,
                                   cfg_range=(0.0, 1.0), num_images_per_prompt=1, generator=generator,
                                   output_type="pil")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            results.images[0].save(out_path)
            print(f"saved {out_path}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED {name}: {e}", flush=True)
    print("B0 batch done", flush=True)


if __name__ == "__main__":
    main()
