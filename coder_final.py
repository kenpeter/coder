"""Single-GPU training for SAN/code model x-small + Muon/ANVIL + torch.compile.
Combines: x-small architecture (D=512, 6 layers), Muon optimizer, full torch.compile.
Target: 40-60k tok/s on RTX 4070 Ti 12 GB.
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

# ── Architecture x-small ─────────────────────────────────────────────────────
NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 4
HEAD_DIM = 128
NARROW_HEAD_DIM = 64
MLP_HIDDEN_DIM = 1024
VOCAB_SIZE = 49152
NGRAM_VOCAB_SIZE = 256_000
NGRAM_DIM = 512
BLOCK_SIZE = 128

NO_ATTN_LAYERS = (1, 3)
NO_MLP_LAYERS = ()
ATTN_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in NO_ATTN_LAYERS)
WIDE_QK_LAYERS = (0, 5)
HALF_V_LAYERS = (1, 4)
LONG_WINDOW_LAYERS = (0, 5)
VALUE_EMBED_LAYERS = (0, 1, 4, 5)

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_TOKENS = 8192
GRAD_ACCUM = 4
MAX_SEQ_LEN = 8192
NUM_STEPS = 1_000_000
LOG_EVERY = 25
SAVE_EVERY_MIN = 30

# ── Muon optimizer (Newton-Schulz orthogonalization, single-GPU) ─────────────

@torch.compile(dynamic=False, fullgraph=False)
def newton_schulz_orthogonalize(G, steps=5):
    """Newton-Schulz iteration to orthogonalize a matrix."""
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = 3 * torch.eye(A.size(0), device=A.device, dtype=A.dtype) - A @ A
        X = X @ B
    return X


class MuonSingle(torch.optim.Optimizer):
    """Simplified Muon for single GPU. Newton-Schulz on 2D weight matrices."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim == 2 and g.size(0) == g.size(1):
                    # Newton-Schulz orthogonalization (square matrices only)
                    g = newton_schulz_orthogonalize(g, steps=ns_steps)
                    g = g * (g.size(0) ** 0.5)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                if nesterov:
                    g = g.lerp_(buf, momentum)
                else:
                    g = buf
                p.add_(g, alpha=-lr)
        return loss


# ── Data loader ──────────────────────────────────────────────────────────────

class ShardDataset(IterableDataset):
    def __init__(self, shard_dirs, batch_tokens, max_seq_len):
        self.shard_dirs = shard_dirs
        self.batch_tokens = batch_tokens
        self.max_seq_len = max_seq_len
        self.shards = []
        for d in shard_dirs:
            self.shards.extend(sorted(glob.glob(os.path.join(d, "*.bin"))))
        self.shard_idx = 0
        self.tokens = None
        self.pos = 0
        self._load_shard(0)

    def _load_shard(self, idx):
        self.shard_idx = idx % len(self.shards)
        path = self.shards[self.shard_idx]
        with open(path, "rb") as f:
            data = f.read()
        self.tokens = torch.frombuffer(data, dtype=torch.uint16).clone()
        self.pos = 0

    def _next_shard(self):
        self._load_shard(self.shard_idx + 1)

    def __iter__(self):
        while True:
            num_tokens_local = self.batch_tokens + 1
            if self.pos + num_tokens_local > self.tokens.numel():
                self._next_shard()
                continue
            buf = self.tokens[self.pos:self.pos + num_tokens_local]
            self.pos += num_tokens_local
            inputs = buf[:-1].to(torch.int32)
            targets = buf[1:].to(torch.int64)
            _cum = torch.tensor([num_tokens_local - 1], dtype=torch.int32)
            yield (inputs, targets, _cum)


def make_loader(shard_dirs, batch_tokens, max_seq_len, num_workers=4):
    ds = ShardDataset(shard_dirs, batch_tokens, max_seq_len)
    return DataLoader(ds, batch_size=None, num_workers=num_workers, persistent_workers=True)


# ── Model ────────────────────────────────────────────────────────────────────

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        num_pairs = self.head_dim // 2
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=num_pairs, dtype=torch.float32)
        self.register_buffer("factor1", torch.empty(self.max_seq_len, num_pairs), persistent=False)
        self.register_buffer("factor2", torch.empty(self.max_seq_len, num_pairs), persistent=False)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        self.factor1.copy_(theta.cos())
        self.factor2.copy_(theta.sin())

    def apply(self, x, cos, sin):
        rotary_dim = 64
        num_pairs = rotary_dim // 2
        x_rope = x[..., :rotary_dim]
        x_rest = x[..., rotary_dim:]
        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        c = cos[..., :num_pairs].view(1, cos.size(0), 1, num_pairs)  # [1,T,1,32]
        s = sin[..., :num_pairs].view(1, sin.size(0), 1, num_pairs)  # [1,T,1,32]
        y1 = x1 * c - x2 * s
        y2 = x1 * s + x2 * c
        x_rope = torch.stack((y1, y2), dim=-1).flatten(-2)
        return torch.cat((x_rope, x_rest), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, qk_dim, v_dim, layer_idx, max_seq_len, device):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.hdim = num_heads * head_dim
        self.layer_idx = layer_idx

        self.qkvo_w = nn.Parameter(torch.empty(dim * 4, self.hdim, device=device, dtype=torch.bfloat16))
        self.attn_gate = nn.Linear(dim, num_heads, bias=False, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.qkvo_w.normal_(0, dim ** -0.5)
            self.attn_gate.weight.zero_()

        self.yarn = Yarn(head_dim, max_seq_len)

    @torch.compile(mode="default", dynamic=False, fullgraph=False)
    def forward(self, x, seqlens, bm_size, key_offset, attn_scale):
        # x: [T, D] -> unsqueeze to [1, T, D] for attention
        x = x.unsqueeze(0)
        B, T = x.size(0), x.size(1)
        qkv = F.linear(x, self.qkvo_w[:self.hdim * 3].type_as(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        pos = torch.zeros(T, dtype=torch.long, device=x.device)
        prev = 0
        for i, end in enumerate(seqlens.tolist()):
            end = int(end)
            pos[prev:end] = torch.arange(end - prev, device=x.device)
            prev = end

        cos = self.yarn.factor1[:T]
        sin = self.yarn.factor2[:T]
        q = self.yarn.apply(q, cos, sin).to(x.dtype)
        k = self.yarn.apply(k, cos, sin).to(x.dtype)

        v_dim = self.v_dim
        if v_dim < self.head_dim:
            v = v.view(B, T, self.num_heads, self.head_dim)[:, :, :, :v_dim].contiguous()

        arange = torch.arange(T, device=x.device)
        doc_starts = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seqlens[:-1]])
        doc_ends = seqlens
        causal = arange[:, None] >= arange[None, :]
        same_doc = (arange[:, None] >= doc_starts[:, None]) & (arange[None, :] < doc_ends[:, None])
        mask = causal & same_doc
        if bm_size is not None and bm_size < T:
            window = arange[:, None] - arange[None, :] < bm_size
            mask = mask & window
        attn_mask = ~mask

        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=attn_mask, is_causal=False, scale=attn_scale)
        y = y.transpose(1, 2).contiguous()

        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y * gate
        y = y.contiguous().view(B, T, self.num_heads * self.v_dim)

        o_w = self.qkvo_w[self.hdim * 3:]
        if self.v_dim < self.head_dim:
            o_w = o_w.view(self.hdim, self.num_heads, self.head_dim)[:, :, :self.v_dim].reshape(self.hdim, self.num_heads * self.v_dim)
        y = F.linear(y, o_w.type_as(y))
        return y.squeeze(0)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, device):
        super().__init__()
        self.c_fc = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=torch.bfloat16))
        self.c_proj = nn.Parameter(torch.empty(hidden_dim, dim, device=device, dtype=torch.bfloat16))
        with torch.no_grad():
            self.c_fc.uniform_(-0.5 * dim ** -0.5 * (3 ** 0.5), 0.5 * dim ** -0.5 * (3 ** 0.5))
            self.c_proj.zero_()

    @torch.compile(mode="default", dynamic=False, fullgraph=False)
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

    @torch.compile(mode="default", dynamic=False, fullgraph=False)
    def forward(self, x, seqlens, bm_size, key_offset, attn_scale, value_embeds, ve_idx, resid_lambda, post_lambda):
        if self.has_attn:
            ve = value_embeds[ve_idx] if ve_idx is not None else None
            ve_gate = self.ve_gate
            attn_out = self.attn(x, seqlens, bm_size, key_offset, attn_scale)
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


def next_multiple_of_n(v, n=128):
    return (v + n - 1) // n * n


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
    print(f"Config: x-small {NUM_LAYERS} layers, D={D_MODEL}, heads={NUM_HEADS}, ngram={NGRAM_VOCAB_SIZE}")
    print(f"Batch: {BATCH_TOKENS} tokens × {GRAD_ACCUM} grad accum = {BATCH_TOKENS * GRAD_ACCUM} effective")
    print("Optimizer: Muon (Newton-Schulz)")
    print("Compile: torch.compile mode=default on all layers")

    model = GPT(VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, HEAD_DIM, D_MODEL, MAX_SEQ_LEN, device)
    model = model.to(device)

    # Full torch.compile on all layers
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
    model.embed = compile_layer(model.embed)
    model.lm_head = compile_layer(model.lm_head)

    # Separate params for Muon vs Adam
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in name and "lm_head" not in name and "ngram" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_opt = MuonSingle(muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
    adam_opt = torch.optim.AdamW(adam_params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < 100:
            return step / 100
        progress = min(1.0, (step - 100) / (NUM_STEPS - 100))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    muon_scheduler = torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
    adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, lr_lambda)

    ckpt_mgr = CheckpointManager(model, muon_opt, muon_scheduler, step=0, device=device,
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

    mtp_weights_schedule = []
    for step in range(NUM_STEPS + 1):
        if step < 100:
            mtp_weights_schedule.append(torch.tensor([1.0, 0.0, 0.0]))
        elif step < 10000:
            mtp_weights_schedule.append(torch.tensor([0.5, 0.3, 0.2]))
        else:
            mtp_weights_schedule.append(torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    model.train()
    step = ckpt_mgr.step
    t0 = time.perf_counter()
    tokens_seen = 0

    print(f"Starting x-small+Muon+compile training at step {step}, target {NUM_STEPS} steps", flush=True)

    for step in range(step, NUM_STEPS):
        try:
            inputs, targets, seqlens = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            inputs, targets, seqlens = next(iter_loader)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        seqlens = seqlens.to(device, non_blocking=True)

        if (step + 1) % GRAD_ACCUM == 1:
            muon_opt.zero_grad(set_to_none=False)
            adam_opt.zero_grad(set_to_none=False)

        loss = model(inputs, targets, seqlens)
        loss = loss / GRAD_ACCUM

        loss.backward()

        tokens_seen += inputs.numel()

        if (step + 1) % GRAD_ACCUM == 0:
            muon_opt.step()
            adam_opt.step()
            muon_scheduler.step()
            adam_scheduler.step()

        ckpt_mgr.step = step + 1
        ckpt_mgr.maybe_save()

        if (step + 1) % LOG_EVERY == 0:
            t1 = time.perf_counter()
            toks_per_sec = tokens_seen / (t1 - t0)
            print(f"step:{step+1}/{NUM_STEPS} loss:{loss.item():.4f} "
                  f"tok/s:{toks_per_sec:.0f} muon_lr:{muon_scheduler.get_last_lr()[0]:.2e} "
                  f"adam_lr:{adam_scheduler.get_last_lr()[0]:.2e}", flush=True)
            tokens_seen = 0
            t0 = t1

    ckpt_mgr.save()
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
