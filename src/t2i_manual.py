import argparse, os
import torch
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from diffusers import AutoencoderKL
# 必须用 OmniGen2 自带调度器：支持 dynamic_time_shift + num_tokens（diffusers 原生版本会忽略，导致出图为纯噪声）
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pretrained_models")
    parser.add_argument("--instruction", default="The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.")
    parser.add_argument("--output_image_path", default="outputs/output_t2i.png")
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
    print("loading mllm...", flush=True)
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(root, "mllm"), torch_dtype=dtype)
    print("loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(os.path.join(root, "mllm_processor"))
    pipeline = OmniGen2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler, mllm=mllm, processor=processor)
    accelerator = Accelerator(mixed_precision="bf16")
    pipeline = pipeline.to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(0)
    results = pipeline(prompt=args.instruction, input_images=None, width=1024, height=1024, num_inference_steps=args.num_inference_step, max_sequence_length=1024, text_guidance_scale=args.text_guidance_scale, image_guidance_scale=2.0, cfg_range=(0.0, 1.0), num_images_per_prompt=1, generator=generator, output_type="pil")
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)
    for i, image in enumerate(results.images):
        name, ext = os.path.splitext(args.output_image_path)
        image.save(f"{name}_{i}{ext}" if len(results.images) > 1 else args.output_image_path)
    print("saved " + args.output_image_path, flush=True)

if __name__ == "__main__":
    main()