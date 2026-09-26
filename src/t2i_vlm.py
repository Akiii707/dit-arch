import argparse, os
import torch
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusers import AutoencoderKL


class AlignMLP(torch.nn.Module):
    # 与 train_align.py 保持一致的推理用结构
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
    # 把 E2(Qwen3.5-VL, hidden 2048) 包装成 OmniGen2 pipeline 期望的 mllm 接口：
    # mllm(input_ids, attention_mask=..., output_hidden_states=True).hidden_states[-1]
    # align_mlp 存在时：hidden 先经过 MLP 对齐到 Qwen2.5-VL 嵌入分布再喂 DiT
    def __init__(self, model, align_mlp=None):
        super().__init__()
        self.model = model
        self.align_mlp = align_mlp

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        # DiffusionPipeline 的 _execution_device/self.device 会访问组件的 .device，
        # transformers 模型自带该属性，普通 nn.Module wrapper 必须补上，否则
        # AttributeError 会被 __getattr__ 掩盖成 "no attribute '_execution_device'"
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        # ForConditionalGeneration 返回的 hidden_states 是逐层文本 hidden，[-1] 为最后一层 (B, L, 2048)
        if output_hidden_states and self.align_mlp is not None:
            h = out.hidden_states[-1]
            # MLP 以 fp32 运行（训练时也是 fp32），输入对齐 dtype 后再转回
            mlp_dtype = next(self.align_mlp.parameters()).dtype
            h = self.align_mlp(h.to(mlp_dtype)).to(h.dtype)
            out.hidden_states = tuple(out.hidden_states[:-1]) + (h,)
        elif output_hidden_states:
            out.hidden_states = tuple(out.hidden_states)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2/pretrained_models")
    parser.add_argument("--vlm_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_visual_VICreg/outputs_e/E2_vicreg/checkpoints/step_5000")
    parser.add_argument("--instruction", default="The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.")
    parser.add_argument("--output_image_path", default="/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i/output_t2i_e2vlm.png")
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    args = parser.parse_args()
    root = os.path.abspath(args.model_path)
    dtype = torch.bfloat16

    print("loading scheduler...", flush=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    print("loading vae...", flush=True)
    vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    print("loading transformer...", flush=True)
    transformer = OmniGen2Transformer2DModel.from_pretrained(root, subfolder="transformer", torch_dtype=dtype)
    print("loading E2 VLM (Qwen3.5-VL)...", flush=True)
    mllm = E2VLMWrapper(
        Qwen3_5ForConditionalGeneration.from_pretrained(args.vlm_path, torch_dtype=dtype)
    ).eval()
    align_dir = "/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/align"
    align_path = f"{align_dir}/align_mlp_real.pt"
    if not os.path.exists(align_path):
        align_path = f"{align_dir}/align_mlp.pt"
    if os.path.exists(align_path):
        mlp = AlignMLP()
        mlp.load_state_dict(torch.load(align_path, map_location="cpu"))
        # 保持 fp32；pipeline.to(device) 会搬到 GPU
        mllm.align_mlp = mlp.eval()
        print(f"loaded align_mlp from {align_path} (fp32)", flush=True)
    print("loading processor (tokenizer replaced by E2)...", flush=True)
    processor = AutoProcessor.from_pretrained(os.path.join(root, "mllm_processor"))
    processor.tokenizer = AutoTokenizer.from_pretrained(args.vlm_path)

    pipeline = OmniGen2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor)
    accelerator = Accelerator(mixed_precision="bf16")
    pipeline = pipeline.to(accelerator.device)
    for p in mllm.parameters():
        p.requires_grad_(False)
    generator = torch.Generator(device=accelerator.device).manual_seed(0)

    with torch.no_grad():
        results = pipeline(prompt=args.instruction, input_images=None, width=1024, height=1024,
                           num_inference_steps=args.num_inference_step, max_sequence_length=1024,
                           text_guidance_scale=args.text_guidance_scale, image_guidance_scale=2.0,
                           cfg_range=(0.0, 1.0), num_images_per_prompt=1, generator=generator,
                           output_type="pil")
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
    for i, image in enumerate(results.images):
        name, ext = os.path.splitext(args.output_image_path)
        image.save(f"{name}_{i}{ext}" if len(results.images) > 1 else args.output_image_path)
    print("saved " + args.output_image_path, flush=True)


if __name__ == "__main__":
    main()