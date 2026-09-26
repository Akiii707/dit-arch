#!/bin/bash
# Original OmniGen2: Qwen2.5-VL + 原版 DiT，无 AlignMLP，无 LoRA
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/t2i_geneval_orig_enh_enh_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== env ==="
hostname
nvidia-smi || echo "WARN: nvidia-smi failed"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

pip uninstall -y torchao 2>/dev/null || true

TF_WHEELS="/data/juicefs_sharing_data/11195663/vlm_visual_ar/vendor/wheels_tf"
if ! python3 -c "import transformers; v=transformers.__version__; assert tuple(int(x) for x in v.split('.')[:2]) >= (5,14), v" 2>/dev/null; then
    echo "[deps] upgrading transformers from offline wheels"
    pip install --upgrade --no-index --find-links "$TF_WHEELS" transformers 2>&1 | tail -3 || true
fi
python3 -c "import transformers; print('[deps] transformers', transformers.__version__)" || true

DIT_WHEELS="$PROJECT_DIR/vendor/wheels_dit"
if ! python3 -c "import diffusers; assert diffusers.__version__ == '0.39.0', diffusers.__version__" 2>/dev/null; then
    echo "[deps] installing diffusers 0.39.0 from offline wheels"
    pip install --no-index --find-links "$DIT_WHEELS" diffusers==0.39.0 2>&1 | tail -3 || true
fi
python3 -c "import diffusers; print('[deps] diffusers', diffusers.__version__)" || true

pip install --no-index --find-links "$DIT_WHEELS" einops omegaconf timm 2>&1 | tail -3 || true
python3 -c 'import diffusers, einops, omegaconf; print("[deps] diffusers", diffusers.__version__, "einops/omegaconf ok")'

echo "=== t2i inference (original OmniGen2, GenEval) ==="

cd "$OMNIGEN2_DIR"
PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" \
python3 -u t2i_orig.py \
    --model_path "$OMNIGEN2_DIR/pretrained_models" \
    --prompts_file "$PROJECT_DIR/v_launch/geneval_prompts_enhanced.txt" \
    --output_dir "$PROJECT_DIR/outputs/t2i/geneval_orig_enh" \
    --num_inference_step 50 \
    --text_guidance_scale 4.0

echo "=== DONE ==="
