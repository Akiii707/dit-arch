import argparse, os
import torch
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from diffusers import AutoencoderKL
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def load_prompts(prompts_file):
    """Load prompts from name|prompt format file."""
    items = []
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, prompt = line.split("|", 1)
            items.append((name, prompt))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pretrained_models")
    parser.add_argument("--instruction", default=None, help="Single prompt mode")
    parser.add_argument("--prompts_file", default=None, help="Batch mode: name|prompt per line")
    parser.add_argument("--output_image_path", default="outputs/output_t2i.png")
    parser.add_argument("--output_dir", default=None, help="Output directory for batch mode")
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    args = parser.parse_args()

    root = os.path.abspath(args.model_path)
    dtype = torch.bfloat16

    print("loading scheduler...", flush=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    print("loading vae...", flush=True)
    vae = AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=dtype)
    print("loading transformer (original)...", flush=True)
    transformer = OmniGen2Transformer2DModel.from_pretrained(root, subfolder="transformer", torch_dtype=dtype)
    print("loading mllm (original Qwen2.5-VL)...", flush=True)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(root, "mllm"), torch_dtype=dtype)
    print("loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(os.path.join(root, "mllm_processor"))

    pipeline = OmniGen2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor)
    accelerator = Accelerator(mixed_precision="bf16")
    pipeline = pipeline.to(accelerator.device)

    # Determine prompts
    if args.prompts_file:
        items = load_prompts(args.prompts_file)
        output_dir = args.output_dir or os.path.join("outputs", "t2i_orig")
        os.makedirs(output_dir, exist_ok=True)
        print(f"=== Original OmniGen2 batch mode: {len(items)} prompts -> {output_dir} ===", flush=True)
        for idx, (name, prompt) in enumerate(items):
            out_path = os.path.join(output_dir, f"{name}.png")
            if os.path.exists(out_path):
                print(f"[{idx+1}/{len(items)}] SKIP {name} (exists)", flush=True)
                continue
            generator = torch.Generator(device=accelerator.device).manual_seed(0)
            results = pipeline(
                prompt=prompt, input_images=None, width=1024, height=1024,
                num_inference_steps=args.num_inference_step,
                max_sequence_length=1024,
                text_guidance_scale=args.text_guidance_scale,
                image_guidance_scale=2.0, cfg_range=(0.0, 1.0),
                num_images_per_prompt=1, generator=generator, output_type="pil"
            )
            results.images[0].save(out_path)
            print(f"[{idx+1}/{len(items)}] saved {out_path}", flush=True)
        print(f"=== Done: {len(items)} images in {output_dir} ===", flush=True)
    else:
        prompt = args.instruction or "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens."
        generator = torch.Generator(device=accelerator.device).manual_seed(0)
        results = pipeline(
            prompt=prompt, input_images=None, width=1024, height=1024,
            num_inference_steps=args.num_inference_step,
            max_sequence_length=1024,
            text_guidance_scale=args.text_guidance_scale,
            image_guidance_scale=2.0, cfg_range=(0.0, 1.0),
            num_images_per_prompt=1, generator=generator, output_type="pil"
        )
        os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
        results.images[0].save(args.output_image_path)
        print("saved " + args.output_image_path, flush=True)


if __name__ == "__main__":
    main()
