#!/bin/bash
# E0_sft 消融组：E0_sft -> Qwen2.5-VL AlignMLP 对齐训练
set -x

D=/data/vjuicefs_ai_gpt_wl/public_data/11195663
PROJECT_DIR=$D/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/align_e0sft_$(date +%Y%m%d-%H%M%S).log"
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

if [ ! -d "$D/vlm_visual_VICreg/outputs_e/E0_sft/checkpoints/step_5000" ]; then
    echo "FATAL: E0_sft checkpoint dir not found"
    exit 1
fi
if [ ! -s "$PROJECT_DIR/outputs/align/real_captions.txt" ]; then
    echo "FATAL: real_captions.txt cache missing"
    exit 1
fi

echo "=== align-e0sft training ==="
cd "$D/OmniGen2"
PYTHONPATH="$D/OmniGen2:$PYTHONPATH" torchrun --nproc_per_node=4 \
    "$D/OmniGen2/train_align_e0sft.py" --epochs 4 --bs 8 --lr 1e-4

RC=$?
echo "=== align-e0sft exit code: $RC ==="
ls -la "$PROJECT_DIR/outputs/align_e0sft/" || true
exit $RC
