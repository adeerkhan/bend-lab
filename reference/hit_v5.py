#!/usr/bin/env python3
"""Faithful port of Nicholas0228/HiT-Released, v4: the six corrections.

  reference/.venv/bin/python reference/hit_v4.py --arm hit --steps 1200 --out results/hit-v4/hit

Corrections over v3 (all verified against the release source):
  1. 2D sincos positional embedding added to image tokens (release model.py).
  2. 8 class-context tokens + 1 time token, concatenated before the blocks and
     carried through every stage but NEVER grouped or skipped (model.py).
  3. router init eye + 1e-3/sqrt(width) with fixed seeds 1701/1702 (grouping.py).
  4. exact tree_metadata: DFS order/intervals; differentiable merge replay via
     node_means prefix sums, not per-merge Python slicing (grouping.py).
  5. stage 2 is variable-length over the actual stage-1 counts (VarLenGrouping).
  6. budget loss uses population = sum of actual incoming counts (grouping.py).
Group maps come from the model's own forward(return_aux=True), not a separate
extraction function.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

import numpy as np
from scipy.cluster.hierarchy import _hierarchy
from hit_linkage import batched_linkage, batched_metadata
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from flowers_report import legibility  # noqa: E402
from unet_flowers_eval import spectrum  # noqa: E402

DATA = os.path.expanduser("~/.cache/bend-lab/flowers")
EDGE, PATCH, GRID = 64, 8, 8
TOKENS, PDIM = GRID * GRID, PATCH * PATCH
AMIN, LAMBDA, TR, TB = 0.05, 0.01, 0.05, 0.1
CTX, ROUTER_SEEDS = 8, (1701, 1702)


def load(n=2048, nh=102):
    tr = np.fromfile(os.path.join(DATA, "flowers64-trainfull-gray.u8"), np.uint8).reshape(-1, EDGE, EDGE)[:n].astype(np.float32) / 255
    ho = np.fromfile(os.path.join(DATA, "flowers64-hold-gray.u8"), np.uint8).reshape(-1, EDGE, EDGE)[:nh].astype(np.float32) / 255
    return tr, ho


def patchify(x):
    B = x.shape[0]
    return x.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, TOKENS, PDIM)


def unpatch(z):
    B = z.shape[0]
    return z.reshape(B, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(B, EDGE, EDGE)


def sincos_pos(d, side=GRID):
    """get_2d_sincos_pos_embed from the release's util/model_util."""
    half = d // 2
    freq = 1 / (10000 ** (np.arange(0, half, 2).astype(np.float32) / half))
    pos = np.arange(side)
    emb = np.einsum('n,f->nf', pos, freq)
    grid = np.concatenate((np.repeat(emb, 2, -1)[None, :, :].repeat(side, 0),
                           np.repeat(emb, 2, -1)[None, :, :].repeat(side, 0).transpose(1, 0, 2)), -1)
    return np.sin(grid).reshape(side * side, d).astype(np.float32)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w)


def modulate(x, sh, sc):
    return x * (1 + sc[:, None]) + sh[:, None]


class SpatialRoPE(nn.Module):
    def __init__(self, head_dim, side=GRID, prefix=0):
        super().__init__(); self.prefix = prefix
        half = head_dim // 2
        freq = 1 / (10000 ** (torch.arange(0, half, 2).float() / half))
        pos = torch.arange(side) / side * side
        phase = torch.einsum('n,f->nf', pos, freq).repeat_interleave(2, -1)
        grid = torch.cat((phase[:, None, :].expand(side, side, -1),
                          phase[None, :, :].expand(side, side, -1)), -1).reshape(side * side, head_dim)
        self.register_buffer('cos', torch.cat((torch.ones(prefix, head_dim), grid.cos())), persistent=False)
        self.register_buffer('sin', torch.cat((torch.zeros(prefix, head_dim), grid.sin())), persistent=False)

    def forward(self, x):
        n = x.shape[2]
        if n > self.cos.shape[0]:
            return x
        cos, sin = self.cos[:n], self.sin[:n]
        paired = x.reshape(*x.shape[:-1], -1, 2)
        rot = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
        return x * cos + rot * sin


class Block(nn.Module):
    def __init__(self, d, heads, adaln=True, rope=None):
        super().__init__()
        self.adaln, self.rope = adaln, rope
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qnorm, self.knorm = RMSNorm(d // heads), RMSNorm(d // heads)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 4 * d, bias=False); self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        self.heads = heads
        if adaln:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
            nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def swiglu(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def attn(self, x):
        B, N, d = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.qnorm(q), self.knorm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, d))

    def forward(self, x, cond=None):
        if self.adaln:
            sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.ada(cond).chunk(6, -1)
            x = x + g_a[:, None] * self.attn(modulate(self.n1(x), sh_a, sc_a))
            return x + g_m[:, None] * self.swiglu(modulate(self.n2(x), sh_m, sc_m))
        return x + self.swiglu(self.attn(self.n1(x)))


class Blocks(nn.Module):
    def __init__(self, n, d, adaln=True, heads=2, rope=False):
        super().__init__()
        r = SpatialRoPE(d // heads, prefix=1) if rope else None
        self.blocks = nn.ModuleList([Block(d, heads, adaln, r) for _ in range(n)])

    def forward(self, x, cond=None):
        for b in self.blocks:
            x = b(x, cond)
        return x


def tree_metadata(children, scores, threshold, n):
    """Exact release grouping.py tree_metadata with DFS order and intervals."""
    parent = np.arange(2 * n - 1, dtype=np.int64)
    for m in range(n - 1):
        if scores[m] >= threshold:
            parent[children[m, 0]] = n + m
            parent[children[m, 1]] = n + m
    roots = np.array([next(i for i in [j] if parent[i] == i) if False else 0 for j in range(n)])
    roots = np.empty(n, np.int64)
    for i in range(n):
        r = i
        while parent[r] != r: r = parent[r]
        roots[i] = r
    lookup = np.full(2 * n - 1, -1, np.int64)
    groups = np.empty(n, np.int64); K = 0
    for i in range(n):
        if lookup[roots[i]] < 0: lookup[roots[i]] = K; K += 1
        groups[i] = lookup[roots[i]]
    order = np.empty(n, np.int64)
    start = np.zeros(2 * n - 1, np.int64); end = np.zeros(2 * n - 1, np.int64)
    stack = [2 * n - 2]; cursor = 0
    while stack:
        node = stack.pop()
        if node < n:
            start[node] = cursor; end[node] = cursor + 1; order[cursor] = node; cursor += 1
        else:
            stack.append(int(children[node - n, 1])); stack.append(int(children[node - n, 0]))
    for m in range(n - 1):
        l, r = int(children[m, 0]), int(children[m, 1])
        start[n + m] = min(start[l], start[r]); end[n + m] = max(end[l], end[r])
    intervals = np.stack((start[:n-1], end[:n-1], start[n:2*n-1], end[n:2*n-1]), -1) if n > 1 else np.zeros((0, 4), np.int64)
    return groups, K, order, intervals


def node_means(vectors, order, intervals):
    """Release node_means: live prefix-sum means in detached DFS order."""
    ordered = vectors.gather(1, order[..., None].expand(-1, -1, vectors.shape[-1]))
    prefix = F.pad(ordered.cumsum(1), (0, 0, 1, 0))
    def mean(s, e):
        lo = prefix.gather(1, s[..., None].expand(-1, -1, prefix.shape[-1]))
        hi = prefix.gather(1, e[..., None].expand(-1, -1, prefix.shape[-1]))
        den = (e - s).clamp_min(1)
        return (hi - lo) / den[..., None]
    return mean(intervals[..., 0], intervals[..., 1]), mean(intervals[..., 2], intervals[..., 3])


class Group(nn.Module):
    def __init__(self, d, seed, ratio=2.0):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                for m in (self.q, self.k):
                    eye = torch.eye(d)
                    m.weight.copy_(eye + torch.randn_like(eye) * (1e-3 / math.sqrt(d)))
        self.skip = nn.Linear(d, d, bias=False)
        with torch.no_grad(): self.skip.weight.copy_(torch.eye(d))
        self.register_buffer('tau', torch.tensor(0.0, dtype=torch.float64))
        self.ratio = ratio

    def project(self, h):
        with torch.autocast(h.device.type, enabled=False):
            x = h.float()
            return F.normalize(self.q(x), dim=-1, eps=1e-6), F.normalize(self.k(x), dim=-1, eps=1e-6)

    def tree(self, q, k, lengths=None):
        B, N, d = q.shape
        S = 0.5 * (q @ k.transpose(-1, -2) + k @ q.transpose(-1, -2))
        Sd = S.detach().cpu().numpy()
        tau = float(self.tau.item())
        hard = np.zeros((B, N), np.int64); counts = np.zeros(B, np.int64)
        orders = []; intervals = []; scores = []
        if lengths is None or np.all(np.asarray(lengths) == N):
            # stage 1: uniform length -> fully batched numba
            trees = batched_linkage(Sd.astype(np.float64))
            g_all, c_all, o_all, iv_all, sc_all = batched_metadata(trees, tau, N)
            hard = g_all; counts = c_all
            scores = [torch.from_numpy(np.ascontiguousarray(sc_all[b], dtype=np.float32)).to(q.device) for b in range(B)]
            orders = [o_all[b] for b in range(B)]
            intervals = [iv_all[b] for b in range(B)]
        else:
          for b in range(B):
            n = max(1, int(lengths[b]))
            iu = np.triu_indices(n, 1)
            dist = np.maximum(0, 1 - Sd[b, :n, :n][iu].astype(np.float64))
            if n <= 1:
                hard[b, :n] = 0; counts[b] = 1
                full_order = np.zeros(N, np.int64); full_order[0] = 0
                full_iv = np.zeros((max(N - 1, 0), 4), np.int64)
                orders.append(full_order); intervals.append(full_iv)
                scores.append(torch.zeros(max(N - 1, 0), device=q.device))
                continue
            if n > 1:
                z = _hierarchy.nn_chain(dist, n, 2)
                sc = 1 - z[:, 2]
                g, K, order, iv = tree_metadata(z[:, :2].astype(np.int64), sc, tau, n)
            else:
                sc = np.zeros(0, np.float64)
                g, K, order, iv = np.zeros(1, np.int64), 1, np.zeros(1, np.int64), np.zeros((0, 4), np.int64)
            hard[b, :n] = g; counts[b] = K
            full_order = np.zeros(N, np.int64); full_order[:n] = order
            full_iv = np.zeros((max(N - 1, 0), 4), np.int64)
            if n > 1: full_iv[:n-1] = iv
            orders.append(full_order); intervals.append(full_iv)
            full_sc = np.zeros(max(N - 1, 0), np.float32)
            if n > 1: full_sc[:n-1] = sc
            scores.append(torch.from_numpy(full_sc).to(q.device))
        return S, torch.from_numpy(hard).to(q.device), torch.from_numpy(counts).to(q.device), \
               torch.from_numpy(np.stack(orders)).to(q.device), \
               torch.from_numpy(np.stack(intervals)).to(q.device), torch.stack(scores)

    def forward(self, h, lengths=None):
        q, k = self.project(h)
        S, hard, counts, order, intervals, scores = self.tree(q, k, lengths)
        B, N, d = h.shape
        K = int(hard.max().item()) + 1
        idx = hard[..., None].expand(-1, -1, d)
        sizes = torch.zeros(B, K, device=h.device).scatter_add_(1, hard, torch.ones(B, N, device=h.device))
        qm = torch.zeros(B, K, d, device=h.device).scatter_add_(1, idx, q)
        km = torch.zeros(B, K, d, device=h.device).scatter_add_(1, idx, k)
        qm /= sizes.clamp_min(1)[..., None]; km /= sizes.clamp_min(1)[..., None]
        logits = 0.5 * (q @ km.transpose(-1, -2) + k @ qm.transpose(-1, -2))
        valid = torch.arange(K, device=h.device)[None, :] < counts[:, None]
        A = (logits / TR).masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        mass = A.sum(1)
        compact = A.transpose(1, 2) @ h / (mass[..., None] + 1e-8)
        ql, qr = node_means(q, order, intervals)
        kl, kr = node_means(k, order, intervals)
        merge = 0.5 * ((ql * kr).sum(-1) + (kl * qr).sum(-1))
        valid_merge = torch.arange(merge.shape[1], device=h.device)[None, :] < (merge.shape[1])
        conf = ((merge - self.tau.float()) / TB).sigmoid()
        n = torch.full((B,), N, device=h.device) if lengths is None else lengths.to(h.device).float()
        soft = n - conf.sum(1)
        # population-based budget (stage 1: B*N; stage 2: actual incoming counts)
        pop = (B * N) if lengths is None else float(lengths.sum())
        fh = counts.float().sum() / pop
        fs = soft.sum() / pop
        comp = self.ratio / (self.ratio - 1) * ((self.ratio - 1) * fh * fs + (1 - fh) * (1 - fs))
        return compact.to(h.dtype), A, comp, hard, counts, order, intervals, scores


class HiT(nn.Module):
    def __init__(self, d=128, depths=(3, 3, 6, 3, 3), heads=2, ctx=CTX):
        super().__init__()
        self.d, self.ctx_n = d, ctx + 1
        self.embed = nn.Linear(PDIM, d)
        self.register_buffer('pos', torch.from_numpy(sincos_pos(d)))
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2701)
            self.register_buffer('ctx_pos', torch.empty(1, ctx, d).normal_(std=0.02))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5801)
            self.register_buffer('time_pos', torch.empty(1, 1, d).normal_(std=0.02))
        e, m, core, md, dec = depths
        self.enc = Blocks(e, d, rope=True)
        self.mid = Blocks(m, d)
        self.core = Blocks(core, d)
        self.dec_mid = Blocks(md, d)
        self.dec = Blocks(dec, d, rope=True)
        self.g1 = Group(d, ROUTER_SEEDS[0]); self.g2 = Group(d, ROUTER_SEEDS[1])
        self.head = nn.Linear(d, PDIM)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        tc = self.t(t.reshape(-1, 1)); yc = self.y(y.reshape(-1, 1))
        cond = tc + yc
        ctx = yc[:, None] + self.ctx_pos
        ctx = torch.cat((tc[:, None] + self.time_pos, ctx), dim=1)
        H = self.embed(patchify(x)) + self.pos.unsqueeze(0)
        X = torch.cat((ctx, H), dim=1)
        X = self.enc(X, cond)
        c, e1 = X[:, :self.ctx_n], X[:, self.ctx_n:]
        r1 = self.g1.skip(e1)
        Z1, A1, c1, hard1, counts1, order1, iv1, sc1 = self.g1(e1)
        X = torch.cat((c, Z1), dim=1); X = self.mid(X, cond)
        e2, c = X[:, self.ctx_n:], X[:, :self.ctx_n]
        r2 = self.g2.skip(e2)
        Z2, A2, c2, hard2, counts2, order2, iv2, sc2 = self.g2(e2, lengths=counts1)
        X = torch.cat((c, Z2), dim=1); X = self.core(X, cond)
        inner, c = X[:, self.ctx_n:], X[:, :self.ctx_n]
        h2 = r2 + A2 @ inner
        X = torch.cat((c, h2), dim=1); X = self.dec_mid(X, cond)
        h2 = X[:, self.ctx_n:]
        h1 = r1 + A1 @ h2
        X = torch.cat((c, h1), dim=1); X = self.dec(X, cond)
        out = X[:, self.ctx_n:]
        pred = unpatch(self.head(out))
        if return_aux:
            return pred, c1 + c2, dict(hard1=hard1, counts1=counts1, A1=A1,
                                       hard2=hard2, counts2=counts2, A2=A2, tau1=self.g1.tau.item(), tau2=self.g2.tau.item())
        return pred, c1 + c2


class Dense(nn.Module):
    """JiT/DiT-style dense arm: same 18 blocks, context tokens, adaLN, rope."""
    def __init__(self, d=128, depth=18, heads=2, ctx=CTX, adaln=True):
        super().__init__()
        self.ctx_n = ctx + 1 if adaln else 0
        self.embed = nn.Linear(PDIM, d)
        self.register_buffer('pos', torch.from_numpy(sincos_pos(d)))
        self.t = nn.Linear(1, d); self.y = nn.Linear(1, d)
        self.register_buffer('ctx_pos', torch.randn(1, ctx, d) * 0.02)
        self.register_buffer('time_pos', torch.randn(1, 1, d) * 0.02)
        self.blocks = Blocks(depth, d, adaln=adaln, rope=True)
        self.head = nn.Linear(d, PDIM)

    def forward(self, x, t, y, return_aux=False):
        B = x.shape[0]
        tc = self.t(t.reshape(-1, 1)); yc = self.y(y.reshape(-1, 1))
        H = self.embed(patchify(x)) + self.pos.unsqueeze(0)
        if self.ctx_n:
            ctx = torch.cat((tc[:, None] + self.time_pos, yc[:, None] + self.ctx_pos), dim=1)
            H = torch.cat((ctx, H), dim=1)
        else:
            H = H + tc[:, None] + yc[:, None]
        H = self.blocks(H, tc + yc if self.ctx_n else None)
        if self.ctx_n: H = H[:, self.ctx_n:]
        pred = unpatch(self.head(H))
        return (pred, H.new_zeros(()), {}) if return_aux else (pred, H.new_zeros(()))


def sample(model, x0, steps=32):
    model.eval()
    x = x0
    with torch.no_grad():
        for t in np.linspace(0, 1, steps + 1)[:-1]:
            tt = torch.full((x.shape[0],), float(t))
            pred, _ = model(x, tt, torch.zeros_like(tt))
            x = x + (pred - x) / max(1 - t, AMIN) * (1 / steps)
    return torch.clamp(x, 0, 1)


def calibrate(model, tr, n=128):
    """Pooled scores over noise levels; per-image population/ratio target."""
    n = min(n, len(tr))
    torch.manual_seed(882014)
    model.eval()
    all1, pop1, all2, pop2 = [], 0, [], 0
    cond = lambda tt: model.t(tt.reshape(-1, 1)) + model.y(torch.zeros(n, 1))
    with torch.no_grad():
        for t in (0.25, 0.5, 0.75, 1.0):
            x = torch.from_numpy(tr[:n]); tt = torch.full((n,), float(t))
            xt = tt.reshape(-1, 1, 1) * x + (1 - t) * torch.randn_like(x)
            H = model.embed(patchify(xt)) + model.pos.unsqueeze(0)
            ctx = torch.cat((model.t(tt.reshape(-1,1))[:, None] + model.time_pos,
                             model.y(torch.zeros(n,1))[:, None] + model.ctx_pos), dim=1)
            X = torch.cat((ctx, H), dim=1); X = model.enc(X, cond(tt))
            e1 = X[:, model.ctx_n:]
            q, k = model.g1.project(e1)
            S = 0.5 * (q @ k.transpose(-1,-2) + k @ q.transpose(-1,-2)).cpu().numpy()
            for b in range(n):
                z = _hierarchy.nn_chain(np.maximum(0, 1 - S[b][np.triu_indices(TOKENS,1)].astype(np.float64)), TOKENS, 2)
                all1.append(1 - z[:, 2]); pop1 += TOKENS
            tau1 = float(np.sort(np.concatenate(all1))[::-1][int(pop1/model.g1.ratio)-1])
            model.g1.tau.fill_(tau1)
            Z1, _, _, _, counts1, _, _, _ = model.g1(e1)
            X = torch.cat((ctx, Z1), dim=1); X = model.mid(X, cond(tt))
            e2 = X[:, model.ctx_n:]
            q, k = model.g2.project(e2)
            S2 = 0.5 * (q @ k.transpose(-1,-2) + k @ q.transpose(-1,-2)).cpu().numpy()
            for b in range(n):
                m = int(counts1[b])
                if m <= 1:
                    continue
                z = _hierarchy.nn_chain(
                    np.maximum(0, 1 - S2[b, :m, :m][np.triu_indices(m,1)].astype(np.float64)), m, 2)
                all2.append(1 - z[:, 2]); pop2 += m
            tau2 = float(np.sort(np.concatenate(all2))[::-1][max(0, int(pop2/model.g2.ratio)-1)]) if all2 else 0.0
            model.g2.tau.fill_(tau2)
    print(f"calibrated tau=({tau1:.3f},{tau2:.3f}) pooled {len(all1)}/{len(all2)} trees", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["hit", "jit", "vit"], required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retune-at", type=int, default=0)

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr, ho = load(args.n_train)
    if args.arm == "hit":
        model = HiT()
        calibrate(model, tr)
    elif args.arm == "jit":
        model = Dense(adaln=True)
    else:
        model = Dense(adaln=False)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    print(f"arm={args.arm} params={npar} steps={args.steps} batch={args.batch} train={len(tr)}", flush=True)
    losses, t0 = [], time.time()
    model.train()
    for step in range(args.steps):
        idx = np.random.randint(0, len(tr), args.batch)
        x = torch.from_numpy(tr[idx])
        tt = torch.rand(args.batch); yy = torch.zeros(args.batch)
        xt = tt.reshape(-1, 1, 1) * x + (1 - tt.reshape(-1, 1, 1)) * torch.randn_like(x)
        pred, comp = model(xt, tt, yy)
        den = torch.clamp(1 - tt, min=AMIN).reshape(-1, 1, 1)
        loss = ((pred - x) ** 2).mean() / (den ** 2).mean() + (LAMBDA * comp if args.arm == "hit" else 0)
        if args.retune_at and step + 1 == args.retune_at:
            calibrate(model, tr)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(1, args.steps // 8) == 0:
            print(f"step {step+1}/{args.steps} loss {losses[-1]:.5f} ({time.time()-t0:.0f}s)", flush=True)

    draws = sample(model, torch.randn(args.draws, EDGE, EDGE)).numpy()
    held = legibility(draws.reshape(args.draws, -1), ho.reshape(-1, EDGE * EDGE))
    trleg = legibility(draws.reshape(args.draws, -1), tr.reshape(-1, EDGE * EDGE))
    ceil = legibility(ho.reshape(-1, EDGE * EDGE), tr.reshape(-1, EDGE * EDGE))
    floor = legibility(np.random.randn(args.draws, EDGE * EDGE).astype(np.float32), tr.reshape(-1, EDGE * EDGE))
    rec = dict(arm=args.arm, paper="arXiv:2611.22962 / Nicholas0228/HiT-Released v4",
               params=npar, steps=args.steps, batch=args.batch, width=128,
               patch=PATCH, tokens=TOKENS, losses=losses,
               first_loss=losses[0], final_loss=losses[-1],
               legibility_held=float(held.mean()), legibility_train=float(trleg.mean()),
               ceiling=float(ceil.mean()), noise_floor=float(floor.mean()),
               spectrum=[float(v) for v in spectrum(draws)],
               wall_seconds=time.time() - t0, draws=args.draws)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    json.dump(rec, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    np.save(os.path.join(args.out, "draws.npy"), draws)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "losses"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
