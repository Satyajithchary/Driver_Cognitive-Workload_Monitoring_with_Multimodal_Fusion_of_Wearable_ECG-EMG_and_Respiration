import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import seed_all
from .data import augment
from .metrics import classification
from .models.encoders import set_snorm_scale
from .models.fusions import build, modalities


def lr_at(ep, frac, t):
    x = ep + frac
    if x < t["warmup_epochs"]:
        return t["lr"] * x / t["warmup_epochs"]
    p = (x - t["warmup_epochs"]) / max(1e-9, t["epochs"] - t["warmup_epochs"])
    return t["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def _groups(data, r, ref, protocol):
    if protocol == "strict":
        return None, None
    g = data.groups(r)
    if protocol == "calib":
        return g, torch.ones(len(r), dtype=torch.bool, device=g.device)
    return g, torch.as_tensor(ref, device=g.device)


@torch.no_grad()
def predict(model, data, r, s, ref, protocol, mods, drop=None):
    """One forward per subject (its scored + reference windows together)."""
    model.eval()
    P, Z, X = [], [], {}
    sub = data.sub[r]
    for u in pd.unique(sub):
        m = sub == u
        x, _ = data.batch(r[m], s[m])
        if drop:
            x[drop] = torch.zeros_like(x[drop])
        g, rf = _groups(data, r[m], ref[m], protocol)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model({k: x[k] for k in mods}, g, rf)
        keep = torch.as_tensor(~ref[m], device=out["logits"].device)
        P.append(F.softmax(out["logits"].float(), -1)[keep].cpu())
        Z.append(out["emb"].float()[keep].cpu())
        for k, v in out.items():
            if k not in ("logits", "emb"):
                v = F.softmax(v.float(), -1) if k == "branch_logits" else v.float()
                X.setdefault(k, []).append(v[keep].cpu())
    return torch.cat(P).numpy(), torch.cat(Z).numpy(), {k: torch.cat(v).numpy() for k, v in X.items()}


def train_run(cfg, data, task, name, protocol, seed, split, out_dir):
    t = cfg["train"]
    tr_s, va_s, te_s = split
    tag = f"{te_s[0]}_seed{seed}"
    out_dir = Path(out_dir)
    ck = out_dir / "checkpoints" / name / f"{tag}.pt"
    pp = out_dir / "preds" / name / f"{tag}.npz"
    if pp.with_suffix(".json").exists():
        return None
    ck.parent.mkdir(parents=True, exist_ok=True)
    pp.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, classes=cfg["tasks"][task]["classes"])
    seed_all(seed * 1000 + int(te_s[0][:4]))
    mods = modalities(name, cfg)
    data.set_protocol(protocol, tr_s)
    set_snorm_scale(protocol != "enroll" or cfg.get("enroll_scale", False))
    W = cfg["window"]
    vr, vs, vf = data.eval_windows(va_s, W["stride_eval_s"])
    yv = data.y[vr[~vf]].cpu().numpy()
    rng = np.random.RandomState(seed * 1000 + int(te_s[0][:4]))
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + int(te_s[0][:4]))

    model = build(name, cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    p_drop = t["modality_dropout"] if len(mods) > 1 else 0.0
    hist, best, best_ep, wait = [], -1.0, -1, 0
    t0 = time.time()
    for ep in range(t["epochs"]):
        model.train()
        batches = list(data.train_batches(tr_s, rng))
        tot = 0.0
        for i, (r, s, ref) in enumerate(batches):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(ep, i / len(batches), t)
            x, y = data.batch(r, s)
            x = augment({k: x[k] for k in mods}, gen, mods, p_drop)
            g, rf = _groups(data, r, ref, protocol)
            w = torch.as_tensor(~ref, device=y.device).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t["amp"]):
                out = model(x, g, rf)
                ce = F.cross_entropy(out["logits"].float(), y, label_smoothing=t["label_smoothing"], reduction="none")
                loss = (ce * w).sum() / w.sum()
                if "branch_logits" in out:
                    for k in range(out["branch_logits"].shape[1]):
                        cb = F.cross_entropy(out["branch_logits"][:, k].float(), y, label_smoothing=t["label_smoothing"],
                                             reduction="none")
                        loss = loss + t["aux_weight"] * (cb * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        pv, _, _ = predict(model, data, vr, vs, vf, protocol, mods)
        m = classification(pv, yv, n_cls=len(cfg["classes"]))
        hist.append(dict(epoch=ep, train_loss=tot / len(batches), val_loss=m["nll"], val_acc=m["acc"], val_f1=m["f1_macro"]))
        if m["f1_macro"] > best + 1e-4:
            best, best_ep, wait = m["f1_macro"], ep, 0
            torch.save(model.state_dict(), ck)
        else:
            wait += 1
            if wait >= t["patience"]:
                break

    model.load_state_dict(torch.load(ck, weights_only=True))
    r, s, ref = data.eval_windows(te_s, W["stride_eval_s"])
    prob, emb, extras = predict(model, data, r, s, ref, protocol, mods)
    keep = ~ref
    save = dict(prob=prob, emb=emb.astype(np.float16), y=data.y[r[keep]].cpu().numpy(), rec=r[keep], start=s[keep],
                **{f"x_{k}": v for k, v in extras.items()})
    if len(mods) > 1:
        for m_ in mods:
            save[f"prob_no_{m_}"] = predict(model, data, r, s, ref, protocol, mods, drop=m_)[0]
    np.savez_compressed(pp, **save)
    pd.DataFrame(hist).to_csv(pp.with_suffix(".hist.csv"), index=False)
    res = classification(prob, save["y"], n_cls=len(cfg["classes"]))
    info = dict(task=task, model=name, protocol=protocol, seed=seed, test_subject=te_s[0], val_subjects=va_s,
                best_epoch=best_ep, epochs_run=len(hist), val_f1=best, test_acc=res["acc"], test_f1=res["f1_macro"],
                params=sum(p.numel() for p in model.parameters()), train_time_s=time.time() - t0)
    pp.with_suffix(".json").write_text(json.dumps(info))
    return info
