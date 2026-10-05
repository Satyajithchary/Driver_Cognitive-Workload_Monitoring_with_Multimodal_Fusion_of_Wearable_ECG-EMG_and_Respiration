import torch
import torch.nn as nn


SNORM_SCALE = True


def set_snorm_scale(flag):
    global SNORM_SCALE
    SNORM_SCALE = bool(flag)


def snorm(x, g, ref, eps=1e-5):
    """Subject normalisation. Each feature is centred (and, if SNORM_SCALE,
    scaled) with the statistics of the *reference* samples (ref=True) sharing
    subject id g, pooled over samples and tokens; applied to every sample of g."""
    B, d = x.shape[0], x.shape[-1]
    f = x.float().reshape(B, -1, d)
    _, inv = torch.unique(g, return_inverse=True)
    G = int(inv.max()) + 1
    w = ref.float()[:, None]
    n = torch.zeros(G, device=x.device).index_add_(0, inv, ref.float() * f.shape[1]).clamp_min(1)
    mu = torch.zeros(G, d, device=x.device).index_add_(0, inv, f.sum(1) * w) / n[:, None]
    c = f - mu[inv][:, None]
    if not SNORM_SCALE:
        return c.reshape(x.shape).to(x.dtype)
    var = torch.zeros(G, d, device=x.device).index_add_(0, inv, (c ** 2).sum(1) * w) / n[:, None]
    return (c / torch.sqrt(var[inv][:, None] + eps)).reshape(x.shape).to(x.dtype)


def block(d, heads, ff_mult, dropout):
    return nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout, activation="gelu",
                                      batch_first=True, norm_first=True)


class SignalTokenizer(nn.Module):
    """1D conv stem + non-overlapping patches: (B, C, T) -> (B, T/patch, d)."""

    def __init__(self, d, ch, patch, k):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(ch, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.GELU(),
        )
        self.patch = nn.Conv1d(32, d, patch, stride=patch)

    def forward(self, x):
        return self.patch(self.stem(x)).transpose(1, 2)


class FrameTokenizer(nn.Module):
    """Per-frame 2D-ViT embedding -> d. (B, T, D) -> (B, T, d)."""

    def __init__(self, d, in_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.proj(x)


class TokenTransformer(nn.Module):
    """CLS + learned positions + pre-norm transformer. Returns final tokens and
    the CLS state after every layer (for hierarchical fusion)."""

    def __init__(self, n_tokens, d, heads, layers, ff_mult, dropout):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_tokens + 1, d) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([block(d, heads, ff_mult, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tok, g=None, ref=None):
        x = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1) + self.pos[:, : tok.shape[1] + 1]
        x = self.drop(x)
        levels = []
        for layer in self.layers:
            x = layer(x)
            levels.append(x[:, 0])
        x = self.norm(x)
        if g is not None:
            x = torch.cat([snorm(x[:, :1], g, ref), snorm(x[:, 1:], g, ref)], 1)
            levels = [snorm(l, g, ref) for l in levels]
        return x, levels


def n_tokens(cfg, mod):
    st = cfg["streams"][mod]
    return cfg["window"]["length_s"] * st["fs"] // st["patch"]


def tokenizer(cfg, mod):
    st = cfg["streams"][mod]
    return SignalTokenizer(cfg["model"]["d"], st["ch"], st["patch"], st["k"])


class ModalityEncoder(nn.Module):
    def __init__(self, cfg, mod):
        super().__init__()
        m = cfg["model"]
        self.tok = tokenizer(cfg, mod)
        self.tf = TokenTransformer(n_tokens(cfg, mod), m["d"], m["heads"], m["layers"], m["ff_mult"], m["dropout"])

    def forward(self, x, g=None, ref=None):
        return self.tf(self.tok(x), g, ref)
