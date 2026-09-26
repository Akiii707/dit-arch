#!/bin/bash
# E2+AlignMLP 条件编码的 OmniGen2 DiT LoRA 微调（4 GPU accelerate FSDP）
set -x

OMNIGEN2_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/OmniGen2
PROJECT_DIR=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/ft_lora_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=== env ==="
hostname
nvidia-smi || echo "WARN: nvidia-smi failed"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

# torchao 与镜像 transformers 冲突，卸载（E 系列 / smoke 已验证）
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
        diffusers==0.39.0 einops omegaconf timm accelerate torchdiffeq 2>&1 | tail -4 || true
    # peft 无离线 wheel，从本地解压包直接复制进 site-packages（0.19.1）
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cp -r /data/vjuicefs_ai_gpt_wl/public_data/11195663/py312_deps/peft "$SITE"/
    cp -r /data/vjuicefs_ai_gpt_wl/public_data/11195663/py312_deps/peft-0.19.1.dist-info "$SITE"/ 2>/dev/null || true
    # train.py 额外依赖 python-dotenv / datasets / matplotlib（容器内缺失），走 vivo 内网镜像安装
    # 模块名->pip 包名映射
    declare -A PKG=( [dotenv]=python-dotenv [datasets]=datasets [matplotlib]=matplotlib [yaml]=pyyaml )
    MISSING=""
    for m in dotenv datasets matplotlib yaml; do
        python3 -c "import $m" 2>/dev/null || MISSING="$MISSING ${PKG[$m]}"
    done
    [ -n "$MISSING" ] && pip install -i http://repo.vivo.lan:8082/repository/proxy-pypi/simple --trusted-host repo.vivo.lan $MISSING 2>&1 | tail -3
fi
python3 -c "import diffusers, einops, omegaconf, peft; print('[deps] diffusers', diffusers.__version__, 'einops/omegaconf/peft ok')" \
    || echo "FATAL: OmniGen2 deps still missing"

# 数据集就绪检查（build_data job 产物）
MIX=/data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i_data/mix.yml
if [ ! -f "$MIX" ]; then
    echo "FATAL: t2i dataset not ready: $MIX missing"
    ls -la /data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i_data/ 2>/dev/null || true
    exit 1
fi
wc -l /data/vjuicefs_ai_gpt_wl/public_data/11195663/vlm_dit_downstream/outputs/t2i_data/t2i_train.jsonl || true

echo "=== LoRA finetune (4 GPU FSDP) ==="
cd "$OMNIGEN2_DIR"
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "visible GPUs: $NGPU"

PYTHONPATH="$OMNIGEN2_DIR:$PYTHONPATH" accelerate launch \
    --num_processes=$NGPU \
    --num_machines=1 \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy HYBRID_SHARD_ZERO2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_forward_prefetch false \
    --fsdp_use_orig_params True \
    --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states True \
    train.py --config options/ft_lora_vlm.yml

RC=$?
echo "=== ft_lora exit code: $RC ==="
ls -la "$OMNIGEN2_DIR/experiments/ft_lora_e2" 2>/dev/null || true
exit $RC