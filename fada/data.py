"""GPU-resident segments of one label task, LOSO splits and the three
calibration protocols:
  strict  no subject information
  enroll  reference = the subject's 5-min rest baseline of the same study (never scored)
  calib   reference = all of the subject's (unlabeled) task segments"""
from pathlib import Path

import numpy as np
import torch

from .common import seg_key, subjects

WAVE_CH = {"ecg": [0, 1], "rsp": [0], "emg": []}   # channels z-scored per window


def load_task(cfg, task):
    t = cfg["tasks"][task]
    d = Path(cfg["paths"]["data"]) / "streams"
    segs, arrays = [], {m: [] for m in cfg["modalities"]}
    for sub in subjects(cfg, task):
        z = np.load(d / f"{sub}.npz")
        items = [(y, s, False) for y, cls in enumerate(t["segments"]) for s in cls] + [(-1, t["reference"], True)]
        for y, (study, phase, level), is_ref in items:
            key = seg_key(study, phase, level)
            for m in cfg["modalities"]:
                arrays[m].append(z[f"{key}|{m}"])
            dur = min(z[f"{key}|{m}"].shape[-1] / cfg["streams"][m]["fs"] for m in cfg["modalities"])
            segs.append(dict(subject=sub, key=key, y=y, ref=is_ref, dur=dur))
    return segs, arrays


def loso_splits(subs, n_val, seed):
    out = []
    for i, s in enumerate(subs):
        rest = [x for x in subs if x != s]
        rng = np.random.RandomState(seed * 1000 + i)
        val = sorted(rng.choice(rest, n_val, replace=False).tolist())
        out.append(([x for x in rest if x not in val], val, [s]))
    return out


class Segments:
    def __init__(self, cfg, segs, arrays, device="cuda"):
        self.cfg, self.dev = cfg, device
        self.L = cfg["window"]["length_s"]
        self.mods = cfg["modalities"]
        self.fs = {m: cfg["streams"][m]["fs"] for m in self.mods}
        self.segs = segs
        self.sub = np.array([s["subject"] for s in segs])
        self.y_np = np.array([s["y"] for s in segs])
        self.is_ref = np.array([s["ref"] for s in segs])
        self.dur = np.array([s["dur"] for s in segs])
        self.X = {}
        for m in self.mods:
            T = max(a.shape[-1] for a in arrays[m])
            buf = np.zeros((len(segs), arrays[m][0].shape[0], T), np.float32)
            for i, a in enumerate(arrays[m]):
                buf[i, :, :a.shape[-1]] = a
            self.X[m] = torch.from_numpy(buf).to(device)
        self.y = torch.as_tensor(np.maximum(self.y_np, 0), device=device)
        self.ar = {m: torch.arange(self.L * self.fs[m], device=device) for m in self.mods}
        self.protocol = "strict"

    # ------------------------------------------------------------ protocol
    def seg_index(self, subs, ref=False):
        return np.where(np.isin(self.sub, subs) & (self.is_ref == ref))[0]

    def set_protocol(self, protocol, train_subs):
        """EMG envelopes carry absolute level -> standardised per protocol."""
        self.protocol = protocol
        X = self.X["emg"]
        C = X.shape[1]
        self.mu = torch.zeros(len(self.segs), C, 1, device=self.dev)
        self.sd = torch.ones(len(self.segs), C, 1, device=self.dev)

        def stats(idx):
            v = torch.cat([X[i, :, : int(self.dur[i] * self.fs["emg"])] for i in idx], 1)
            return v.mean(1, keepdim=True), v.std(1, keepdim=True) + 1e-6

        g_mu, g_sd = stats(self.seg_index(train_subs))
        if protocol == "strict":
            self.mu[:], self.sd[:] = g_mu, g_sd
            return
        for s in np.unique(self.sub):
            idx = np.where(self.sub == s)[0]
            ref = idx[self.is_ref[idx]] if protocol == "enroll" else idx[~self.is_ref[idx]]
            mu, sd = stats(ref)
            if protocol == "enroll" and not self.cfg.get("enroll_scale", False):
                sd = g_sd  # baseline correction: rest mean removed, population scale kept
            self.mu[idx], self.sd[idx] = mu, sd

    # ------------------------------------------------------------ windows
    def _starts(self, i, stride):
        return np.arange(0, self.dur[i] - self.L + 1e-9, stride)

    def eval_windows(self, subs, stride):
        """Scored windows grouped by subject (+ reference windows in enroll)."""
        R, S, F = [], [], []
        for s in subs:
            for i in self.seg_index([s]):
                st = self._starts(i, stride)
                R.append(np.full(len(st), i)); S.append(st); F.append(np.zeros(len(st), bool))
            if self.protocol == "enroll":
                for i in self.seg_index([s], ref=True):
                    st = self._starts(i, 5.0)
                    R.append(np.full(len(st), i)); S.append(st); F.append(np.ones(len(st), bool))
        return np.concatenate(R), np.concatenate(S), np.concatenate(F)

    def train_batches(self, subs, rng):
        T = self.cfg["train"]
        G, B, stride = T["group_size"], T["batch_size"], self.cfg["window"]["stride_train_s"]
        chunks = []
        for s in subs:
            r, st = [], []
            for i in self.seg_index([s]):
                n = int((self.dur[i] - self.L) // stride) + 1
                r.append(np.full(n, i)); st.append(rng.uniform(0, self.dur[i] - self.L, n))
            r, st = np.concatenate(r), np.concatenate(st)
            p = rng.permutation(len(r))
            r, st = r[p], st[p]
            refs = self.seg_index([s], ref=True)
            for a in range(0, len(r), G):
                cr, cs = r[a:a + G], st[a:a + G]
                cf = np.zeros(len(cr), bool)
                if self.protocol == "enroll":
                    i0 = refs[0]
                    cr = np.r_[cr, np.full(8, i0)]
                    cs = np.r_[cs, rng.uniform(0, self.dur[i0] - self.L, 8)]
                    cf = np.r_[cf, np.ones(8, bool)]
                chunks.append((cr, cs, cf))
        order = rng.permutation(len(chunks))
        per = max(1, B // G)
        for a in range(0, len(order), per):
            c = [chunks[j] for j in order[a:a + per]]
            yield tuple(np.concatenate([x[k] for x in c]) for k in range(3))

    def batch(self, r, s):
        r_t = torch.as_tensor(r, device=self.dev)
        s_t = torch.as_tensor(s, device=self.dev, dtype=torch.float32)
        x = {}
        for m in self.mods:
            idx = (s_t * self.fs[m]).round().long()[:, None] + self.ar[m]
            C = self.X[m].shape[1]
            w = self.X[m][r_t[:, None, None], torch.arange(C, device=self.dev)[None, :, None], idx[:, None, :]]
            if m == "emg":
                w = (w - self.mu[r_t]) / self.sd[r_t]
            for c in WAVE_CH[m]:
                v = w[:, c]
                w[:, c] = (v - v.mean(-1, keepdim=True)) / (v.std(-1, keepdim=True) + 1e-6)
            x[m] = w
        return x, self.y[r_t]

    def groups(self, r):
        _, inv = np.unique(self.sub[r], return_inverse=True)
        return torch.as_tensor(inv, device=self.dev)


def augment(x, gen, mods, p_drop=0.0):
    dev = next(iter(x.values())).device
    B = next(iter(x.values())).shape[0]
    out = {}
    for m, v in x.items():
        v = v.clone()
        if m in ("ecg", "rsp"):
            for c in WAVE_CH[m]:
                v[:, c] = v[:, c] * torch.empty(B, 1, device=dev).uniform_(0.8, 1.2, generator=gen)
            v = v + 0.05 * torch.randn(v.shape, device=dev, generator=gen)
        else:
            v = v + 0.1 * torch.randn(v.shape, device=dev, generator=gen)
        out[m] = v
    if p_drop > 0 and len(mods) > 1:
        drop = torch.rand(B, len(mods), device=dev, generator=gen) < p_drop
        allg = drop.all(1)
        if allg.any():
            keep = torch.randint(0, len(mods), (int(allg.sum()),), device=dev, generator=gen)
            drop[allg.nonzero().squeeze(1), keep] = False
        for i, m in enumerate(mods):
            out[m] = out[m] * (~drop[:, i]).float().view(B, 1, 1)
    return out
