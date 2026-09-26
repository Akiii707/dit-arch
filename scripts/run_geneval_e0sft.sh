#!/bin/bash
# B0 对照组：E0_sft + align_mlp_e0sft + 原版 DiT，批量 20 prompt 出图
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/t2i_b0_$(date +%Y%m%d-%H%M%S).log"
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
    echo "[deps] installing OmniGen2 deps from offline wheels: $DIT_WHEELS"
    pip install --no-index --find-links "$DIT_WHEELS" \
        diffusers==0.39.0 einops omegaconf timm accelerate 2>&1 | tail -4 || true
fi
python3 -c "import diffusers, einops, omegaconf; print('[deps] diffusers', diffusers.__version__, 'einops/omegaconf ok')" \
    || echo "FATAL: OmniGen2 deps still missing"

if [ ! -f "$PROJECT_DIR/outputs/align_e0sft/align_mlp_e0sft.pt" ]; then
    echo "FATAL: align_mlp_base.pt not found"
    exit 1
fi

echo "=== B0 inference (base + align_base + original DiT) ==="
cd "$OMNIGEN2_DIR"
PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" python3 -u "$OMNIGEN2_DIR/t2i_vlm_b0.py" \
    --model_path "$OMNIGEN2_DIR/pretrained_models" \
    --vlm_path "/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_visual_VICreg/outputs_e/E0_sft/checkpoints/step_5000" \
    --align_path "$PROJECT_DIR/outputs/align_e0sft/align_mlp_e0sft.pt" \
    --prompts_file "$PROJECT_DIR/v_launch/geneval_prompts.txt" \
    --output_dir "$PROJECT_DIR/outputs/t2i/geneval_e0sft"

RC=$?
echo "=== t2i_b0 exit code: $RC ==="
ls -la "$PROJECT_DIR/outputs/t2i/geneval_e0sft/" || true
exit $RC
