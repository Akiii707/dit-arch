"""Qwen3.5-2B-Base -> Qwen2.5-VL 嵌入对齐训练（E2 的 base 对照组）。

与 train_align.py 唯一差异：源编码器从 E2(vicreg step_5000) 换成 Qwen3.5-2B-Base，
语料/超参/网络结构完全一致，保存 align_mlp_base.pt。
caption 走本地缓存 real_captions.txt（483k 句，无需 OCR 卷挂载）。
"""
import argparse, os, itertools, math, random
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Qwen3_5ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer

D = "/data/vjuicefs_ai_gpt_wl/public_data/11195663"
BASE_PATH = "/data/juicefs_sharing_data/public_data/11195663/model/Qwen3.5-2B-Base"
QWEN_PATH = f"{D}/OmniGen2/pretrained_models/mllm"
SAVE_DIR = f"{D}/vlm_dit_downstream/outputs/align_base"

CAP_CACHE = f"{D}/vlm_dit_downstream/outputs/align/real_captions.txt"


def build_captions(n=30000, seed=0):
    with open(CAP_CACHE) as f:
        caps = [l.rstrip("\n") for l in f if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(caps)
    return caps[:n]


class AlignMLP(nn.Module):
    def __init__(self, dim=2048, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.net(x) * self.scale


@torch.no_grad()
def encode(model, tok, texts, device, max_len=128):
    enc = tok(texts, padding="longest", truncation=True, max_length=max_len, return_tensors="pt").to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False, output_hidden_states=True)
    h = out.hidden_states[-1]
    return h.float(), enc.attention_mask.bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count() or 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        print(f"[align-base] world={world}", flush=True)

    dtype = torch.bfloat16
    print(f"[rank{rank}] loading Qwen3.5-2B-Base ...", flush=True)
    e2 = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_PATH, torch_dtype=dtype).eval().to(device)
    print(f"[rank{rank}] loading Qwen2.5-VL teacher ...", flush=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=dtype).eval().to(device)
    tok_e2 = AutoTokenizer.from_pretrained(BASE_PATH)
    tok_q = AutoTokenizer.from_pretrained(f"{D}/OmniGen2/pretrained_models/mllm_processor")
    for p in itertools.chain(e2.parameters(), qwen.parameters()):
        p.requires_grad_(False)

    mlp = AlignMLP().to(device).float()
    ddp = nn.parallel.DistributedDataParallel(mlp, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.01)

    caps = build_captions()
    n = len(caps)
    print(f"[rank{rank}] captions={n}", flush=True)
    per_rank = math.ceil(n / world)
    my = caps[rank * per_rank:(rank + 1) * per_rank]
    steps_total = args.epochs * math.ceil(len(my) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 100, 0.5 * (1 + math.cos(math.pi * s / max(steps_total, 1)))))

    os.makedirs(SAVE_DIR, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        order = torch.randperm(len(my), generator=torch.Generator().manual_seed(ep)).tolist()
        for i in range(0, len(my) - args.bs + 1, args.bs):
            batch = [my[j] for j in order[i:i + args.bs]]
            he, mask_e = encode(e2, tok_e2, batch, device, args.max_len)
            hq, mask_q = encode(qwen, tok_q, batch, device, args.max_len)
            L = min(he.shape[1], hq.shape[1])
            he, hq = he[:, :L], hq[:, :L]
            m = mask_e[:, :L] & mask_q[:, :L]
            pred = ddp(he)
            se = ((pred - hq) ** 2).masked_fill(~m.unsqueeze(-1), 0.0).sum() / (m.sum() * he.shape[-1])
            pr_c, hq_c = pred[m], hq[m]
            sc = 1.0 - torch.nn.functional.cosine_similarity(pr_c, hq_c, dim=-1).mean()
            loss = se + 0.1 * sc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if rank == 0 and step % 20 == 0:
                with torch.no_grad():
                    hq_c = hq[m]
                    pr_c = pred.detach()[m]
                    cos = torch.nn.functional.cosine_similarity(pr_c, hq_c, dim=-1).mean()
                print(f"ep{ep} step{step}/{steps_total} mse {loss.item():.4f} cos {cos.item():.4f} "
                      f"|teacher| {hq_c.norm(dim=-1).mean().item():.2f}", flush=True)
    if rank == 0:
        torch.save(mlp.state_dict(), f"{SAVE_DIR}/align_mlp_base.pt")
        print(f"[align-base] saved {SAVE_DIR}/align_mlp_base.pt", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
