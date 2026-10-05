import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ModalityEncoder, TokenTransformer, n_tokens, tokenizer


def head(d_in, d, n_cls, p):
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, n_cls))


def split_head(h, x):
    z = h[:3](x)
    return z, h[3:](z)


class Base(nn.Module):
    def __init__(self, cfg, mods):
        super().__init__()
        self.mods = list(mods)
        self.d = cfg["model"]["d"]
        self.p = cfg["model"]["dropout"]
        self.n_cls = len(cfg["classes"])

    def encode(self, x, g, ref):
        hs, lv = [], []
        for m in self.mods:
            h, l = self.enc[m](x[m], g, ref)
            hs.append(h)
            lv.append(l)
        return hs, lv


class _Encoders(Base):
    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        self.enc = nn.ModuleDict({m: ModalityEncoder(cfg, m) for m in self.mods})


class Unimodal(_Encoders):
    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        self.head = head(self.d, self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        (h,), _ = self.encode(x, g, ref)
        z, logits = split_head(self.head, h[:, 0])
        return dict(logits=logits, emb=z)


class EarlyFusion(Base):
    """Low-level tokens of all modalities (+ modality-type embeddings) share one
    transformer from the first layer; depth = sum of the unimodal depths."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        m = cfg["model"]
        self.tok = nn.ModuleDict({k: tokenizer(cfg, k) for k in self.mods})
        self.pos = nn.ParameterDict({k: nn.Parameter(torch.randn(1, n_tokens(cfg, k), self.d) * 0.02) for k in self.mods})
        self.type_emb = nn.Parameter(torch.randn(len(self.mods), self.d) * 0.02)
        N = sum(n_tokens(cfg, k) for k in self.mods)
        self.tf = TokenTransformer(N, self.d, m["heads"], m["layers"] * len(self.mods), m["ff_mult"], m["dropout"])
        nn.init.zeros_(self.tf.pos)
        self.tf.pos.requires_grad_(False)
        self.head = head(self.d, self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        t = [self.tok[k](x[k]) + self.pos[k] + self.type_emb[i] for i, k in enumerate(self.mods)]
        h, _ = self.tf(torch.cat(t, 1), g, ref)
        z, logits = split_head(self.head, h[:, 0])
        return dict(logits=logits, emb=z)


class InterConcat(_Encoders):
    """Concatenated encoder embeddings (torchmultimodal ConcatFusionModule)."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        self.head = head(len(self.mods) * self.d, self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        z, logits = split_head(self.head, torch.cat([h[:, 0] for h in hs], -1))
        return dict(logits=logits, emb=z)


class InterAttention(_Encoders):
    """torchmultimodal AttentionFusionModule: softmax weights over modalities."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        M, d = len(self.mods), self.d
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d)) for _ in range(M)])
        self.attn = nn.Sequential(nn.Linear(M * d, d), nn.Tanh(), nn.Linear(d, M))
        self.head = head(d, d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        f = [h[:, 0] for h in hs]
        w = F.softmax(self.attn(torch.cat(f, -1)), -1)
        fused = sum(w[:, i:i + 1] * self.proj[i](v) for i, v in enumerate(f))
        z, logits = split_head(self.head, fused)
        return dict(logits=logits, emb=z, mod_weights=w)


class LateFusion(_Encoders):
    """Decision level: one classifier per modality, logits mixed with learned
    weights; per-branch losses are added by the trainer."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        self.heads = nn.ModuleList([head(self.d, self.d, self.n_cls, self.p) for _ in self.mods])
        self.w = nn.Parameter(torch.zeros(len(self.mods)))

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        outs = [split_head(hd, h[:, 0]) for hd, h in zip(self.heads, hs)]
        w = F.softmax(self.w, 0)
        logits = sum(w[i] * o[1] for i, o in enumerate(outs))
        res = dict(logits=logits, emb=torch.cat([o[0] for o in outs], -1), mod_weights=w.expand(len(logits), -1))
        res["branch_logits"] = torch.stack([o[1] for o in outs], 1)
        return res


class CrossModalLayer(nn.Module):
    """LXMERT cross-modality layer for M streams: each stream cross-attends to
    the concatenated tokens of the others (cross-attention weights shared, as in
    LXMERT), then its own self-attention and FFN."""

    def __init__(self, M, d, heads, ff_mult, p):
        super().__init__()
        self.x_attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.n_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(M)])
        self.s_attn = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=p, batch_first=True) for _ in range(M)])
        self.n_s = nn.ModuleList([nn.LayerNorm(d) for _ in range(M)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(p), nn.Linear(ff_mult * d, d))
                                 for _ in range(M)])
        self.n_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(M)])
        self.drop = nn.Dropout(p)

    def forward(self, hs):
        q = [n(h) for n, h in zip(self.n_q, hs)]
        new = []
        for i, h in enumerate(hs):
            kv = torch.cat([q[j] for j in range(len(hs)) if j != i], 1)
            new.append(h + self.drop(self.x_attn(q[i], kv, kv, need_weights=False)[0]))
        out = []
        for i, h in enumerate(new):
            s = self.n_s[i](h)
            h = h + self.drop(self.s_attn[i](s, s, s, need_weights=False)[0])
            out.append(h + self.drop(self.ff[i](self.n_f[i](h))))
        return out


class LateCrossAttention(_Encoders):
    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        m = cfg["model"]
        self.x = nn.ModuleList([CrossModalLayer(len(self.mods), self.d, m["heads"], m["ff_mult"], self.p)
                                for _ in range(m["xattn_layers"])])
        self.head = head(len(self.mods) * self.d, self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        for layer in self.x:
            hs = layer(hs)
        z, logits = split_head(self.head, torch.cat([h[:, 0] for h in hs], -1))
        return dict(logits=logits, emb=z)


class TensorFusion(_Encoders):
    """TFN (Zadeh et al. 2017): outer product of [h_m; 1] over all modalities,
    keeping unimodal, bimodal and trimodal terms."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        k = cfg["model"]["tensor_dim"]
        self.pr = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.d), nn.Linear(self.d, k)) for _ in self.mods])
        self.drop = nn.Dropout(self.p)
        self.head = head((k + 1) ** len(self.mods), self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        one = hs[0].new_ones(len(hs[0]), 1)
        t = None
        for pr, h in zip(self.pr, hs):
            z = torch.cat([pr(h[:, 0]), one], -1)
            t = z if t is None else torch.einsum("bi,bj->bij", t, z).flatten(1)
        z, logits = split_head(self.head, self.drop(t))
        return dict(logits=logits, emb=z)


class LowRankFusion(_Encoders):
    """LMF (Liu et al. 2018): rank-r factorisation of the TFN weight tensor."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        r, d = cfg["model"]["lmf_rank"], self.d
        self.fac = nn.ParameterList([nn.Parameter(torch.randn(r, d + 1, d) * (d + 1) ** -0.5) for _ in self.mods])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in self.mods])
        self.rank_w = nn.Parameter(torch.ones(1, r) / r)
        self.bias = nn.Parameter(torch.zeros(d))
        self.head = head(d, d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        one = hs[0].new_ones(len(hs[0]), 1)
        prod = 1.0
        for ln, fac, h in zip(self.ln, self.fac, hs):
            prod = prod * torch.einsum("bi,rio->bro", torch.cat([ln(h[:, 0]), one], -1), fac)
        fused = torch.einsum("xr,bro->bo", self.rank_w, prod) + self.bias
        z, logits = split_head(self.head, fused)
        return dict(logits=logits, emb=z)


class HigherOrderFusion(_Encoders):
    """Polynomial tensor pooling (Hou et al. 2019): order-P CP product of
    z=[1; h_1; ...; h_M], i.e. every interaction up to order P."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        m = cfg["model"]
        k, self.P, self.R = m["tensor_dim"], m["hot_order"], m["hot_rank"]
        self.pr = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.d), nn.Linear(self.d, k)) for _ in self.mods])
        self.factors = nn.ModuleList([nn.Linear(len(self.mods) * k + 1, self.R * self.d, bias=False) for _ in range(self.P)])
        self.norm = nn.LayerNorm(self.d)
        self.head = head(self.d, self.d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        hs, _ = self.encode(x, g, ref)
        zj = torch.cat([hs[0].new_ones(len(hs[0]), 1)] + [pr(h[:, 0]) for pr, h in zip(self.pr, hs)], -1)
        prod = 1.0
        for f in self.factors:
            prod = prod * f(zj).view(-1, self.R, self.d)
        z, logits = split_head(self.head, self.norm(prod.sum(1)))
        return dict(logits=logits, emb=z)


class HierarchicalFusion(_Encoders):
    """Fusion at every encoder depth: per level [CLS_1..CLS_M ; mean pairwise
    products] -> MLP, then attention over levels."""

    def __init__(self, cfg, mods):
        super().__init__(cfg, mods)
        L, d, M = cfg["model"]["layers"], self.d, len(self.mods)
        self.ln = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d) for _ in range(M)]) for _ in range(L)])
        self.level = nn.ModuleList([nn.Sequential(nn.Linear((M + 1) * d, d), nn.GELU(), nn.Dropout(self.p)) for _ in range(L)])
        self.level_emb = nn.Parameter(torch.randn(L, d) * 0.02)
        self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.head = head(d, d, self.n_cls, self.p)

    def forward(self, x, g=None, ref=None):
        _, lv = self.encode(x, g, ref)
        fl = []
        for i in range(len(self.level)):
            c = [self.ln[i][j](lv[j][i]) for j in range(len(self.mods))]
            pw = [a * b for a, b in itertools.combinations(c, 2)] or [c[0] * 0]
            fl.append(self.level[i](torch.cat(c + [torch.stack(pw).mean(0)], -1)) + self.level_emb[i])
        fl = torch.stack(fl, 1)
        w = F.softmax(self.score(fl).squeeze(-1), -1)
        z, logits = split_head(self.head, (w.unsqueeze(-1) * fl).sum(1))
        return dict(logits=logits, emb=z, level_weights=w)


FUSIONS = {
    "early": EarlyFusion, "inter_concat": InterConcat, "inter_attn": InterAttention, "late": LateFusion,
    "late_xattn": LateCrossAttention, "tfn": TensorFusion, "lmf": LowRankFusion, "hot": HigherOrderFusion,
    "hierarchical": HierarchicalFusion,
}


def parse(name):
    """'ecg_only' -> (Unimodal, [ecg]); 'late' -> all streams;
    'late@ecg+rppg' -> late fusion over a subset."""
    if name.endswith("_only"):
        return Unimodal, [name[:-5]]
    if "@" in name:
        f, mods = name.split("@")
        return FUSIONS[f], mods.split("+")
    return FUSIONS[name], None


def build(name, cfg):
    cls, mods = parse(name)
    return cls(cfg, mods or cfg["modalities"])


def modalities(name, cfg):
    return parse(name)[1] or cfg["modalities"]
