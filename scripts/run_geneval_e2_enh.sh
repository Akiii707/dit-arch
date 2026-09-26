#!/bin/bash
# OmniGen2 t2i 推理（LoRA 微调后）：加载 merge_lora.py 合并后的 transformer 权重
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream
FT_CKPT=$OMNIGEN2_DIR/experiments/ft_lora_e2/checkpoint-2500

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/t2i_ft_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== env ==="
hostname
nvidia-smi || echo "WARN: nvidia-smi failed"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# torchao 与镜像 transformers 冲突，卸载
pip uninstall -y torchao 2>/dev/null || true

# transformers>=5.14 离线 wheels
TF_WHEELS="/data/juicefs_sharing_data/11195663/vlm_visual_ar/vendor/wheels_tf"
if ! python3 -c "import transformers; v=transformers.__version__; assert tuple(int(x) for x in v.split('.')[:2]) >= (5,14), v" 2>/dev/null; then
    echo "[deps] upgrading transformers from offline wheels"
    pip install --upgrade --no-index --find-links "$TF_WHEELS" transformers 2>&1 | tail -3 || true
fi
python3 -c "import transformers; print('[deps] transformers', transformers.__version__)" || true

# OmniGen2 依赖离线 wheels（diffusers 必须 0.39.0）
DIT_WHEELS="$PROJECT_DIR/vendor/wheels_dit"
if ! python3 -c "import diffusers; assert diffusers.__version__ == '0.39.0', diffusers.__version__" 2>/dev/null; then
    echo "[deps] installing OmniGen2 deps from offline wheels: $DIT_WHEELS"
    pip install --no-index --find-links "$DIT_WHEELS" \
        diffusers==0.39.0 einops omegaconf timm accelerate 2>&1 | tail -4 || true
fi
python3 -c "import diffusers, einops, omegaconf; print('[deps] diffusers', diffusers.__version__, 'einops/omegaconf ok')" \
    || echo "FATAL: OmniGen2 deps still missing"

# 合并后的 transformer 权重必须在
if [ ! -f "$FT_CKPT/transformer_merged.bin" ]; then
    echo "FATAL: merged transformer weights not found at $FT_CKPT/transformer_merged.bin"
    exit 1
fi

echo "=== t2i inference (finetuned, batch 20) ==="
cd "$OMNIGEN2_DIR"
PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" python3 -u "$OMNIGEN2_DIR/t2i_vlm_ft.py" \
    --model_path "$OMNIGEN2_DIR/pretrained_models" \
    --transformer_weights "$FT_CKPT/transformer_merged.bin" \
    --prompts_file "$PROJECT_DIR/v_launch/geneval_prompts_enhanced.txt" \
    --output_dir "$PROJECT_DIR/outputs/t2i/geneval_e2_lora_enh"
RC=$?
echo "=== t2i_ft batch exit code: $RC ==="
exit $RC
