"""Single-GPU training for SAN/code model (modded-nanogpt #360 architecture).
Fits on RTX 4070 Ti 12 GB at 150 W. Loads _shards_code_*.bin uint16 shards.
Checkpoint every 30 min, overwriting previous (replace_checkpoint=True).
Resume from trainer_state/trainer_state.pt.
"""
import glob
import math
import os
import signal
import sys
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# ── Architecture (from abstraction.md / modded-nanogpt #360) ─────────────────

NUM_LAYERS = 11
D_MODEL = 768
NUM_HEADS = 6
HEAD_DIM = 128          # model_dim / num_heads
NARROW_HEAD_DIM = 64
MLP_HIDDEN_DIM = 2816
VOCAB_SIZE = 49152      # StarCoder2
NGRAM_VOCAB_SIZE = 256_000   # shrunk from 2M to free ~2.6 GB VRAM for bigger batch
NGRAM_DIM = 768
BLOCK_SIZE = 128

# Attention topology
NO_ATTN_LAYERS = (4, 6, 7, 9)
NO_MLP_LAYERS = (7,)
ATTN_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in NO_ATTN_LAYERS)
WIDE_QK_LAYERS = (3, 10)
HALF_V_LAYERS = (1, 8)
LONG_WINDOW_LAYERS = (3, 10)
VALUE_EMBED_LAYERS = (1, 2, 8, 10)

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_TOKENS = 6144
GRAD_ACCUM = 4            # effective batch = 24,576 tokens/step
MAX_SEQ_LEN = 8192   # must exceed max document length (batch_tokens)
LR = 1e-3
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
NUM_STEPS = 1_000_000
LOG_EVERY = 25
SAVE_EVERY_MIN = 30

# ── Data ─────────────────────────────────────────────────────────────────────

BOS_ID = 50256
HEADER_BYTES = 256 * 4


def next_multiple_of_n(v, n=128):
    return math.ceil(v / n) * n


def load_shard(path):
    """Load raw uint16 .bin shard (no header)."""
    with open(path, "rb") as f:
        data = f.read()
    tokens = torch.frombuffer(data, dtype=torch.uint16).clone()
    return tokens


class ShardDataset(IterableDataset):
    """BOS-aligned varlen batches from uint16 .bin shards."""
    def __init__(self, shard_dirs, batch_tokens, max_seq_len):
        self.shard_dirs = shard_dirs
        self.batch_tokens = batch_tokens
        self.max_seq_len = max_seq_len
        self.files = []
        for d in shard_dirs:
            self.files.extend(sorted(glob.glob(os.path.join(d, "*.bin"))))
        assert self.files, f"no shards found in {shard_dirs}"
        self.file_idx = 0
        self.tokens = load_shard(self.files[self.file_idx])
        self.pos = 0

    def _next_shard(self):
        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.file_idx = 0
        self.tokens = load_shard(self.files[self.file_idx])
        self.pos = 0

    def __iter__(self):
        while True:
            num_tokens_local = self.batch_tokens + 1  # +1 for target shift
            if self.pos + num_tokens_local > self.tokens.numel():
                self._next_shard()
                continue
            buf = self.tokens[self.pos:self.pos + num_tokens_local]
            self.pos += num_tokens_local
            inputs = buf[:-1].to(torch.int32)
            targets = buf[1:].to(torch.int64)
            # Single document: cum_seqlens = [num_tokens_local]
            _cum = torch.tensor([num_tokens_local - 1], dtype=torch.int32)
            yield (inputs, targets, _cum)


def make_loader(shard_dirs, batch_tokens, max_seq_len):
    dataset = ShardDataset(shard_dirs, batch_tokens, max_seq_len)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    return loader


# ── Model ────────────────────────────────────────────────────────────────────

class YarnRoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False, attn_scale=0.1, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        # RoPE applies to first 64 dims; factor tables are num_pairs wide (interleaved pairs)
        num_pairs = 32
        self.register_buffer("factor1", torch.empty(max_seq_len, num_pairs, device=device, dtype=torch.bfloat16))
        self.register_buffer("factor2", torch.empty(max_seq_len, num_pairs, device=device, dtype=torch.bfloat16))
        self.reset()

    def reset(self):
        rotary_dim = 64
        num_pairs = rotary_dim // 2  # 32 pairs for interleaved RoPE
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=num_pairs, dtype=torch.float32)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        # factor tables: [max_seq_len, num_pairs] — one freq per interleaved pair
        self.factor1.copy_(theta.cos().to(torch.bfloat16))
        self.factor2.copy_(theta.sin().to(torch.bfloat16))

    def apply(self, x, cos, sin):
        # x: [B, T, H, head_dim], RoPE applied to first 64 dims in interleaved pairs
        rotary_dim = 64
        x_rope = x[..., :rotary_dim]
        x_rest = x[..., rotary_dim:]
        # Interleaved pairs: even indices (cos), odd indices (sin)
        x1 = x_rope[..., 0::2]  # [B, T, H, 32]
        x2 = x_rope[..., 1::2]  # [B, T, H, 32]
        cos = cos[None, :, None, :]  # [1, T, 1, 32]
        sin = sin[None, :, None, :]  # [1, T, 1, 32]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        x_rope_out = torch.empty_like(x_rope)
        x_rope_out[..., 0::2] = y1
        x_rope_out[..., 1::2] = y2
        if x_rest.numel() > 0:
            return torch.cat([x_rope_out, x_rest], dim=-1)
        return x_rope_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, qk_dim, v_dim, layer_idx, max_seq_len, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.hdim = num_heads * head_dim

        std = dim ** -0.5
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(dim * 4, self.hdim, device=device, dtype=torch.bfloat16))
        with torch.no_grad():
            self.qkvo_w[:dim * 3].uniform_(-bound, bound)
            self.qkvo_w[dim * 3:].zero_()

        self.attn_gate = nn.Linear(12, num_heads, bias=False, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.attn_gate.weight.zero_()

        wide = layer_idx in WIDE_QK_LAYERS
        self.yarn = YarnRoPE(qk_dim, max_seq_len,
                             paired=layer_idx in (0, 2, 5),
                             attn_scale=0.085 if wide else 0.13,
                             device=device)

    def forward(self, x, seqlens, bm_size, key_offset, attn_scale, ve=None, ve_gate=None):
        # x: [T, D] packed varlen (B=1)
        T = x.size(0)
        x = x.unsqueeze(0)  # [1, T, D]
        B = 1
        qkv = F.linear(x, self.qkvo_w[:self.hdim * 3].type_as(x))
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).chunk(3, dim=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Position ids restart at each document boundary (varlen packing)
        pos = torch.zeros(T, dtype=torch.long, device=x.device)
        prev = 0
        for end in seqlens.tolist():
            end = int(end)
            pos[prev:end] = torch.arange(end - prev, device=x.device)
            prev = end
        cos = self.yarn.factor1[pos]
        sin = self.yarn.factor2[pos]
        q = self.yarn.apply(q, cos, sin)
        k = self.yarn.apply(k, cos, sin)

        if ve is not None and ve_gate is not None:
            ve_gate_out = 2 * torch.sigmoid(ve_gate(x[..., :ve_gate.weight.size(-1)]))
            v = v + ve_gate_out.unsqueeze(-1) * ve.view(B, T, self.num_heads, self.head_dim)

        if key_offset and T > 1:
            k[:, 1:, :, self.head_dim // 4:self.head_dim // 2] = k[:, :-1, :, self.head_dim // 4:self.head_dim // 2]
            k[:, 1:, :, 3 * self.head_dim // 4:] = k[:, :-1, :, 3 * self.head_dim // 4:]

        v_dim = self.v_dim
        if v_dim < self.head_dim:
            v = v.view(B, T, self.num_heads, self.head_dim)[:, :, :, :v_dim].contiguous()

        # Build attention mask from seqlens (document boundaries + sliding window)
        # Vectorized: no Python loop over docs
        arange = torch.arange(T, device=x.device)
        doc_starts = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seqlens[:-1]])
        doc_ends = seqlens
        # causal mask: j <= i
        causal = arange[:, None] >= arange[None, :]
        # doc boundary mask: same doc
        same_doc = (arange[:, None] >= doc_starts[:, None]) & (arange[None, :] < doc_ends[:, None])
        mask = causal & same_doc
        if bm_size is not None and bm_size < T:
            window = arange[:, None] - arange[None, :] < bm_size
            mask = mask & window
        attn_mask = ~mask

        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=attn_mask, is_causal=False,
                                           scale=attn_scale)
        y = y.transpose(1, 2).contiguous()  # [B, T, H, v_dim]

        # Gate
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y * gate  # broadcast over v_dim
        y = y.contiguous().view(B, T, self.num_heads * self.v_dim)

        # Output projection: slice O columns for HALF_V layers
        o_w = self.qkvo_w[self.hdim * 3:]  # [hdim, hdim]
        if self.v_dim < self.head_dim:
            o_w = o_w.view(self.hdim, self.num_heads, self.head_dim)[:, :, :self.v_dim].reshape(self.hdim, self.num_heads * self.v_dim)
        y = F.linear(y, o_w.type_as(y))
        return y.squeeze(0)  # [T, D]


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, device):
        super().__init__()
        self.c_fc = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=torch.bfloat16))
        self.c_proj = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=torch.bfloat16))
        with torch.no_grad():
            self.c_fc.uniform_(-0.5 * dim ** -0.5 * (3 ** 0.5), 0.5 * dim ** -0.5 * (3 ** 0.5))
            self.c_proj.zero_()

    def forward(self, x):
        x = F.linear(x, self.c_fc.type_as(x))
        x = F.relu(x).square()
        x = F.linear(x, self.c_proj.T.type_as(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim, layer_idx, max_seq_len, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_attn = layer_idx not in NO_ATTN_LAYERS
        self.has_mlp = layer_idx not in NO_MLP_LAYERS

        if self.has_attn:
            qk_dim = HEAD_DIM if layer_idx in WIDE_QK_LAYERS else NARROW_HEAD_DIM
            v_dim = NARROW_HEAD_DIM if layer_idx in HALF_V_LAYERS else HEAD_DIM
            self.attn = Attention(dim, num_heads, head_dim, qk_dim, v_dim, layer_idx, max_seq_len, device)
        if self.has_mlp:
            self.mlp = MLP(dim, MLP_HIDDEN_DIM, device)

        if layer_idx in VALUE_EMBED_LAYERS:
            self.ve_gate = nn.Linear(6, num_heads, bias=False, device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                self.ve_gate.weight.zero_()
        else:
            self.ve_gate = None

    def forward(self, x, seqlens, bm_size, key_offset, attn_scale, value_embeds, ve_idx, resid_lambda, post_lambda):
        if self.has_attn:
            ve = value_embeds[ve_idx] if ve_idx is not None else None
            ve_gate = self.ve_gate
            attn_out = self.attn(x, seqlens, bm_size, key_offset, attn_scale, ve=ve, ve_gate=ve_gate)
            x = x + resid_lambda * attn_out

        if self.has_mlp:
            mlp_in = F.rms_norm(x, (x.size(-1),))
            mlp_out = self.mlp(mlp_in)
            x = x + post_lambda * mlp_out

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim, max_seq_len, device):
        super().__init__()
        self.num_layers = num_layers
        self.device = device
        self.vocab_size = next_multiple_of_n(vocab_size, 128)

        self.register_buffer("ngram_table", torch.zeros(NGRAM_VOCAB_SIZE, NGRAM_DIM, device=device, dtype=torch.bfloat16))
        self.register_buffer("ngram_sign_pool", (torch.randn(8192, NGRAM_DIM).sign()).to(torch.bfloat16))

        self.embed = nn.Embedding(self.vocab_size, model_dim, device=device, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(model_dim, self.vocab_size, bias=False, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.embed.weight)

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, head_dim, i, max_seq_len, device)
            for i in range(num_layers)
        ])

        self.mudd_w1 = nn.Parameter(torch.empty(2, 64, model_dim, device=device, dtype=torch.bfloat16))
        self.mudd_w2 = nn.Parameter(torch.zeros(2, 14, 64, device=device, dtype=torch.bfloat16))
        self.mudd_b2 = nn.Parameter(torch.zeros(2, 14, device=device, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.mudd_w1[0], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mudd_w1[1], a=math.sqrt(5))

        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1 ** 0.5, device=device))
        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2, device=device))
        self.smear_gate = nn.Linear(6, 1, bias=False, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.smear_gate.weight.zero_()

        num_ve = len(VALUE_EMBED_LAYERS)
        self.value_embeds = nn.Parameter(0.01 * torch.randn(num_ve * self.vocab_size, model_dim, device=device, dtype=torch.bfloat16))

    def forward(self, input_seq, target_seq, seqlens, mtp_weights=None, prefix_weight=0.0):
        assert input_seq.ndim == 1
        T = input_seq.size(0)

        x = self.embed(input_seq)

        smear_gate_out = torch.sigmoid(self.smear_gate(x[1:, :6]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])

        x0 = F.rms_norm(x[None], (x.size(-1),))[0]

        bigram_slots = self._ngram_slots(input_seq, "bigram")
        trigram_slots = self._ngram_slots(input_seq, "trigram")
        bigram_rows = self.ngram_table[bigram_slots] * self.ngram_sign_pool[bigram_slots % 8192]
        trigram_rows = self.ngram_table[trigram_slots] * self.ngram_sign_pool[trigram_slots % 8192]
        x0_bigram = (bigram_rows + trigram_rows) / 2

        ve_planes = self.value_embeds.view(len(VALUE_EMBED_LAYERS), self.vocab_size, -1)[:, input_seq]

        cache = {0: x}
        ve_idx = 0
        for i in range(self.num_layers):
            ve = ve_planes[ve_idx] if i in VALUE_EMBED_LAYERS else None
            if ve is not None:
                ve_idx += 1

            bm_size = BLOCK_SIZE * (3 if i in LONG_WINDOW_LAYERS else 1)
            key_offset = i in LONG_WINDOW_LAYERS
            attn_scale = 0.085 if i in WIDE_QK_LAYERS else 0.13

            x = self.blocks[i](x, seqlens, bm_size, key_offset, attn_scale,
                               ve, None, self.resid_lambdas[i, 0], self.post_lambdas[i, 1])
            cache[i] = x

        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)

        if mtp_weights is not None and mtp_weights.size(0) > 1:
            n_predict = mtp_weights.size(0)
            logits_flat = logits.view(-1, logits.size(-1))
            idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)
            target_logits = logits_flat.gather(1, idx)
            cross_entropy = torch.logsumexp(logits_flat, dim=-1).unsqueeze(1) - target_logits
            for k in range(1, n_predict):
                cross_entropy[-k:, k] = 0
            loss = (cross_entropy * mtp_weights).sum()
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum")

        return loss

    def _ngram_slots(self, tokens, channel):
        T = tokens.size(0)
        if channel == "bigram":
            slots = torch.zeros(T, dtype=torch.int64, device=tokens.device)
            if T > 1:
                slots[1:] = (tokens[1:] * 36313 + tokens[:-1] * 27191) % (NGRAM_VOCAB_SIZE // 2)
            return slots
        else:
            slots = torch.zeros(T, dtype=torch.int64, device=tokens.device)
            if T > 2:
                slots[2:] = (tokens[2:] * 17351 + tokens[1:-1] * 60961 + tokens[:-2] * 45259) % (NGRAM_VOCAB_SIZE // 2) + NGRAM_VOCAB_SIZE // 2
            return slots


# ── Checkpointing ────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, model, optimizer, scheduler, step, device,
                 ckpt_path="san_latest.pt",
                 state_dir="trainer_state", save_every_min=30):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = step
        self.device = device
        self.ckpt_path = ckpt_path
        self.state_dir = state_dir
        self.save_every_sec = save_every_min * 60
        self.last_save = time.time()
        os.makedirs(state_dir, exist_ok=True)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _sigterm_handler(self, signum, frame):
        print(f"\nSIGTERM received at step {self.step}, saving checkpoint...", flush=True)
        self.save()
        sys.exit(0)

    def maybe_save(self):
        if time.time() - self.last_save >= self.save_every_sec:
            self.save()
            self.last_save = time.time()

    def save(self):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else {},
            "rng": torch.get_rng_state(),
        }
        torch.save(ckpt, self.ckpt_path)
        state = {
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        }
        torch.save(state, os.path.join(self.state_dir, "trainer_state.pt"))
        print(f"Checkpoint saved at step {self.step}: {self.ckpt_path}", flush=True)

    def load(self, path=None):
        path = path or self.ckpt_path
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        rng = ckpt["rng"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.to(torch.uint8)
        torch.set_rng_state(rng.cpu())
        print(f"Resumed from {path} at step {self.step}", flush=True)
        return True

    def load_trainer_state(self, path=None):
        path = path or os.path.join(self.state_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        rng = state["rng"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.to(torch.uint8)
        torch.set_rng_state(rng.cpu())
        print(f"Resumed from trainer state at step {self.step}", flush=True)
        return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = GPT(VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, HEAD_DIM, D_MODEL, MAX_SEQ_LEN, device)
    model = model.to(device)

    # torch.compile mode="default" — no CUDAGraphs (avoids grad accumulation conflict).
    # Compiles each block's forward separately for speed.
    import torch._dynamo as torchdynamo
    torchdynamo.config.suppress_errors = True

    def compile_layer(layer):
        try:
            return torch.compile(layer, mode="default", dynamic=False, fullgraph=False)
        except Exception:
            return layer

    model.blocks = nn.ModuleList([compile_layer(b) for b in model.blocks])
    for i, b in enumerate(model.blocks):
        if hasattr(b, 'attn'):
            b.attn = compile_layer(b.attn)
        if hasattr(b, 'mlp'):
            b.mlp = compile_layer(b.mlp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < 100:
            return step / 100
        progress = min(1.0, (step - 100) / (NUM_STEPS - 100))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_mgr = CheckpointManager(model, optimizer, scheduler, step=0, device=device,
                                  save_every_min=SAVE_EVERY_MIN)

    if not ckpt_mgr.load_trainer_state():
        ckpt_mgr.load()

    shard_dirs = [
        "/mnt/file_drive/data/_shards_code_easy",
        "/mnt/file_drive/data/_shards_code_medium",
        "/mnt/file_drive/data/_shards_code_hard",
    ]
    loader = make_loader(shard_dirs, BATCH_TOKENS, MAX_SEQ_LEN)
    iter_loader = iter(loader)

    # MTP weights schedule
    mtp_weights_schedule = []
    for step in range(NUM_STEPS + 1):
        x = step / (NUM_STEPS / 4)
        if x < 1 / 3:
            w = [1.0, 0.5, 0.25 * (1 - 3 * x)]
        elif x < 2 / 3:
            w = [1.0, 0.5 * (1 - (3 * x - 1))]
        else:
            w = [1.0]
        mtp_weights_schedule.append(torch.tensor(w, device=device))

    model.train()
    step = ckpt_mgr.step
    t0 = time.perf_counter()
    tokens_seen = 0

    print(f"Starting training at step {step}, target {NUM_STEPS} steps", flush=True)

    for step in range(step, NUM_STEPS):
        try:
            inputs, targets, seqlens = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            inputs, targets, seqlens = next(iter_loader)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        seqlens = seqlens.to(device, non_blocking=True)

        mtp_weights = mtp_weights_schedule[step]

        # Zero grads at start of accum cycle for torch.compile + CUDA graphs.
        # set_to_none=False keeps grad buffer addresses stable across cycles.
        if (step + 1) % GRAD_ACCUM == 1:
            optimizer.zero_grad(set_to_none=False)

        loss = model(inputs, targets, seqlens, mtp_weights=mtp_weights)
        loss = loss / GRAD_ACCUM

        loss.backward()

        tokens_seen += inputs.numel()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        ckpt_mgr.step = step + 1
        ckpt_mgr.maybe_save()

        if (step + 1) % LOG_EVERY == 0:
            t1 = time.perf_counter()
            toks_per_sec = tokens_seen / (t1 - t0)
            print(f"step:{step+1}/{NUM_STEPS} loss:{loss.item():.4f} "
                  f"tok/s:{toks_per_sec:.0f} lr:{scheduler.get_last_lr()[0]:.2e}", flush=True)
            tokens_seen = 0
            t0 = t1

    ckpt_mgr.save()
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()