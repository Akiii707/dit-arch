#!/bin/bash
# E2-VLM -> Qwen2.5-VL 嵌入对齐训练（4 GPU torchrun DDP）
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/align_train_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== env ==="
hostname
nvidia-smi || echo "WARN: nvidia-smi failed"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# torchao 与镜像 transformers 冲突，卸载（E 系列 / smoke 已验证）
pip uninstall -y torchao 2>/dev/null || true

# transformers>=5.14 离线 wheels
TF_WHEELS="/data/juicefs_sharing_data/11195663/vlm_visual_ar/vendor/wheels_tf"
if ! python3 -c "import transformers; v=transformers.__version__; assert tuple(int(x) for x in v.split('.')[:2]) >= (5,14), v" 2>/dev/null; then
    echo "[deps] upgrading transformers from offline wheels"
    pip install --upgrade --no-index --find-links "$TF_WHEELS" transformers 2>&1 | tail -3 || true
fi
python3 -c "import transformers; print('[deps] transformers', transformers.__version__)" || true

# OmniGen2 依赖离线 wheels
# diffusers 必须 0.39.0：0.33.1 与 transformers 5.14.1 不兼容（FLAX_WEIGHTS_NAME 已移除）
DIT_WHEELS="$PROJECT_DIR/vendor/wheels_dit"
if ! python3 -c "import diffusers, diffusers.utils; assert diffusers.__version__ == '0.39.0', diffusers.__version__" 2>/dev/null; then
    echo "[deps] installing OmniGen2 deps from offline wheels: $DIT_WHEELS"
    pip install --no-index --find-links "$DIT_WHEELS" \
        diffusers==0.39.0 einops omegaconf timm accelerate 2>&1 | tail -4 || true
fi
python3 -c "import diffusers, einops, omegaconf; print('[deps] diffusers', diffusers.__version__, 'einops/omegaconf ok')" \
    || echo "FATAL: OmniGen2 deps still missing"

echo "=== align training (4 GPU DDP) ==="
cd "$OMNIGEN2_DIR"
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "visible GPUs: $NGPU"
PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" torchrun --nproc_per_node="$NGPU" \
    --master_port=29517 "$OMNIGEN2_DIR/train_align.py" \
    --epochs 3 --bs 8 --lr 1e-4

RC=$?
echo "=== align train exit code: $RC ==="
exit $RC
