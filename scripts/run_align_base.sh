#!/bin/bash
# Qwen3.5-2B-Base -> Qwen2.5-VL 嵌入对齐训练（E2 的 base 对照组，4 GPU torchrun DDP）
# 语料/超参与 train_align.py 完全一致，仅源编码器不同
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/align_base_$(date +%Y%m%d-%H%M%S).log"
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

# caption 缓存必须存在（避免依赖 OCR 卷挂载）
CAP_CACHE="$PROJECT_DIR/outputs/align/real_captions.txt"
if [ ! -f "$CAP_CACHE" ]; then
    echo "FATAL: caption cache not found at $CAP_CACHE"
    exit 1
fi
wc -l "$CAP_CACHE"

echo "=== align-base training (4 GPU DDP) ==="
cd "$OMNIGEN2_DIR"
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "visible GPUs: $NGPU"
PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" torchrun --nproc_per_node="$NGPU" \
    --master_port=29519 "$OMNIGEN2_DIR/train_align_base.py" \
    --epochs 3 --bs 8 --lr 1e-4

RC=$?
echo "=== align-base train exit code: $RC ==="
ls -la "$PROJECT_DIR/outputs/align_base/" || true
exit $RC
