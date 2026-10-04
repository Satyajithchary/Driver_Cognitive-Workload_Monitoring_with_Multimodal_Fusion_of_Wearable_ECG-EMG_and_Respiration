"""Aggregates the LOSO runs of one ADABase task and protocol.

Per seed, all held-out subjects are pooled; tables report mean ± SD over
seeds at three decision scales: 20-s window, 1-min (mean of the windows that
start in each minute) and session (the whole 10-min recording).
Statistics use the subject as the unit (per-subject scores averaged
over seeds): Friedman + Kendall's W, Nemenyi CD, Holm-corrected Wilcoxon with
rank-biserial r, subject bootstrap CIs, Wilcoxon vs chance."""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sst
from sklearn.manifold import TSNE
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg
from fada import metrics as M
from fada import stats as S
from fada.models.fusions import build, modalities
from fada.plotting import (SLOTS, FAMILY_COLORS, GRAY, INK2, SEQ_CMAP, family, label, save, short, style)

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--protocol", required=True)
ap.add_argument("--runs", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--models", nargs="*")
args = ap.parse_args()

cfg = load_cfg()
CL = cfg["tasks"][args.task]["classes"]
NC = len(CL)
CCOL = SLOTS[:NC]
cfg["classes"] = CL
RUNS = Path(args.runs or Path(cfg["paths"]["results"]) / "runs") / args.task / args.protocol
TAG = args.tag or f"{args.task}/{args.protocol}"
OUT = Path(cfg["paths"]["results"]) / "analysis" / TAG
FIG = Path(cfg["paths"]["figures"]) / TAG
OUT.mkdir(parents=True, exist_ok=True)
style()


def load_runs():
    runs = {}
    names = args.models or [p.name for p in sorted((RUNS / "preds").glob("*"))]
    order = cfg["unimodal"] + cfg["fusions"]
    names = sorted(names, key=lambda n: (order.index(n) if n in order else 99, n))
    for name in names:
        for f in sorted((RUNS / "preds" / name).glob("*_seed*.npz")):
            sub, seed = f.stem.split("_seed")
            z = dict(np.load(f))
            z["sid"] = np.full(len(z["y"]), sub)
            z["info"] = json.loads(f.with_suffix(".json").read_text())
            z["hist"] = pd.read_csv(f.with_suffix(".hist.csv"))
            runs.setdefault(name, {}).setdefault(int(seed), {})[sub] = z
    return runs


RUN = load_runs()
MODELS = list(RUN)
n_sub = {m: {s: len(v) for s, v in RUN[m].items()} for m in MODELS}
print("runs:", {m: sum(v.values()) for m, v in n_sub.items()})


def pooled(rs, key="prob"):
    subs = sorted(rs)
    cat = lambda k: np.concatenate([rs[s][k] for s in subs])
    return cat(key), cat("y"), cat("sid"), cat("rec"), cat("start")


def aggregate(P, y, rec, start, scale):
    """Mean probability per decision unit: window | minute | session."""
    if scale == "window":
        return P, y
    key = rec * 100 + (start // 60).astype(int) if scale == "minute" else rec
    u = np.unique(key)
    return (np.stack([P[key == k].mean(0) for k in u]), np.array([y[key == k][0] for k in u]))


# ----------------------------------------------------------------- tables
KEYS = ["acc", "bal_acc", "f1_macro", "kappa", "mcc", "auroc", "ece"]
rows = []
for m in MODELS:
    for seed, rs in RUN[m].items():
        P, y, sid, rec, st = pooled(rs)
        for scale in ("window", "minute", "session"):
            Pa, ya = aggregate(P, y, rec, st, scale)
            rows.append(dict(model=m, seed=seed, scale=scale, n=len(ya), **M.classification(Pa, ya, n_cls=NC)))
met = pd.DataFrame(rows)
met.to_csv(OUT / "metrics_per_seed.csv", index=False)


def mean_sd(df, keys):
    g = df.groupby("model", sort=False)[keys]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


summary = []
for scale in ("window", "minute", "session"):
    t = mean_sd(met[met.scale == scale], KEYS)
    t.to_csv(OUT / f"table_{scale}.csv")
    print(f"\n{scale.upper()} level (mean ± sd over seeds)\n", t[["acc", "f1_macro", "kappa", "auroc"]].to_string())
    s = met[met.scale == scale].groupby("model", sort=False)[["acc", "f1_macro", "kappa", "auroc"]].mean()
    summary.append(s.add_prefix(f"{scale}_"))
pd.concat(summary, axis=1).loc[MODELS].to_csv(OUT / "summary_means.csv")
cls_keys = [f"{p}_{k}" for k in range(NC) for p in ("prec", "rec", "f1")]
mean_sd(met[met.scale == "window"], cls_keys).to_csv(OUT / "table_per_class.csv")

# --------------------------------------------------------- subject-level
def per_subject(fn):
    out = {}
    for m in MODELS:
        per = []
        for seed, rs in RUN[m].items():
            P, y, sid, _, _ = pooled(rs)
            per.append({s: fn(P[sid == s], y[sid == s]) for s in np.unique(sid)})
        out[m] = pd.DataFrame(per).mean()
    return pd.DataFrame(out)[MODELS]


subj_f1 = per_subject(lambda P, y: M.classification(P, y, n_cls=NC)["f1_macro"])
subj_acc = per_subject(lambda P, y: (P.argmax(1) == y).mean())
subj_f1.to_csv(OUT / "subject_f1.csv")
subj_acc.to_csv(OUT / "subject_acc.csv")

fr = S.friedman(subj_f1.values)
ranks = S.mean_ranks(subj_f1.values)
cd = S.nemenyi_cd(len(MODELS), len(subj_f1))
ci = []
for i, m in enumerate(MODELS):
    mu, lo, hi = S.bootstrap_ci(subj_f1[m].values)
    ci.append(dict(model=m, f1_subject_mean=mu, ci_lo=lo, ci_hi=hi, mean_rank=ranks[i], acc_subject_mean=subj_acc[m].mean(),
                   p_gt_chance=sst.wilcoxon(subj_acc[m] - 1 / NC, alternative="greater").pvalue))
ci = pd.DataFrame(ci)
ci["p_gt_chance_holm"] = multipletests(ci.p_gt_chance, method="holm")[1]
ci.to_csv(OUT / "subject_level.csv", index=False)
pd.DataFrame([dict(**fr, nemenyi_cd=cd)]).to_csv(OUT / "friedman.csv", index=False)
print(f"\nFriedman (blocks = {len(subj_f1)} subjects): chi2={fr['chi2']:.2f} p={fr['p']:.3g} W={fr['kendall_w']:.3f} CD={cd:.2f}")
print(ci.round(3).to_string())

uni = [m for m in MODELS if m.endswith("_only")]
fus = [m for m in MODELS if m not in uni]
best_uni = max(uni, key=lambda m: subj_f1[m].mean()) if uni else None
if best_uni and fus:
    pw = S.pairwise_wilcoxon(subj_f1, pairs=[(m, best_uni) for m in fus])
    pw.to_csv(OUT / "wilcoxon_vs_best_unimodal.csv", index=False)
    print(f"\nfusion vs best unimodal ({best_uni}), Holm\n", pw.round(4).to_string())
S.pairwise_wilcoxon(subj_f1).to_csv(OUT / "wilcoxon_all_pairs.csv", index=False)
best = ci.sort_values("f1_subject_mean").model.iloc[-1]
json.dump({"best_model": best, "best_unimodal": best_uni}, open(OUT / "best.json", "w"))

# ---------------------------------------------------------------- figures
def bar_f1():
    sc = met.groupby(["model", "scale"], sort=False).f1_macro.mean().unstack()
    fig, ax = plt.subplots(figsize=(max(6, 0.62 * len(MODELS)), 3.2))
    x = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        r = ci[ci.model == m].iloc[0]
        c = FAMILY_COLORS[family(m)]
        ax.bar(i, sc.loc[m, "window"], 0.62, color=c, zorder=2)
        ax.scatter(i, sc.loc[m, "session"], marker="D", s=22, c="white", edgecolor=INK2, zorder=4, lw=0.9)
        sv = met[(met.model == m) & (met.scale == "window")].f1_macro.values
        ax.errorbar(i, sc.loc[m, "window"], yerr=sv.std() if len(sv) > 1 else 0, c=INK2, lw=1, capsize=3, zorder=3)
    ax.axhline(1 / NC, c=GRAY, ls="--", lw=0.8)
    ax.set_xticks(x, [short(m) for m in MODELS], rotation=30 if len(MODELS) > 12 else 0, fontsize=7.5)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    hand = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    hand.append(plt.Line2D([], [], marker="D", ls="", mfc="white", mec=INK2))
    ax.legend(hand, list(FAMILY_COLORS) + ["session level"], ncol=7, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.14))
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "f1_by_model")


def cd_diagram():
    k = len(MODELS)
    r = dict(zip(MODELS, ranks))
    srt = sorted(MODELS, key=lambda m: r[m])
    fig, ax = plt.subplots(figsize=(7.2, 0.6 + 0.2 * k))
    ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([1, k], [0.85, 0.85], c=INK2, lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [0.85, 0.87], c=INK2, lw=1)
        ax.text(t, 0.89, str(t), ha="center", fontsize=6.5, color=INK2)
    ax.plot([1, 1 + cd], [0.97, 0.97], c="#e34948", lw=2)
    ax.text(1 + cd / 2, 0.985, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    half = (k + 1) // 2
    step = 0.72 / max(half, 1)
    for i, m in enumerate(srt):
        left = i < half
        y = 0.75 - step * (i if left else i - half)
        xe = 0.7 if left else k + 0.3
        ax.plot([r[m], r[m], xe], [0.85, y, y], c=FAMILY_COLORS[family(m)] if family(m) != "Unimodal" else INK2, lw=1)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{short(m)} ({r[m]:.2f})", ha="right" if left else "left", va="center", fontsize=6.5)
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and r[srt[j + 1]] - r[srt[i]] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for n, (i, j) in enumerate(groups):
        ax.plot([r[srt[i]] - 0.03, r[srt[j]] + 0.03], [0.80 - 0.025 * n] * 2, c="0.15", lw=2.2)
    fig.suptitle(f"Mean rank, subject macro-F1 (Friedman p = {fr['p']:.2g}, n = {len(subj_f1)})", fontsize=8, y=1.04)
    save(fig, FIG / "cd_diagram")


def confusion_grid(models):
    cols = 4
    rows_ = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(2.3 * cols, 2.2 * rows_))
    for a in np.atleast_1d(axes).flat[len(models):]:
        a.axis("off")
    for a, m in zip(np.atleast_1d(axes).flat, models):
        C = sum(M.confusion(*pooled(rs)[:2], n_cls=NC) for rs in RUN[m].values()).astype(float)
        C /= C.sum(1, keepdims=True)
        a.imshow(C, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(NC):
            for j in range(NC):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if C[i, j] > 0.55 else "black")
        a.set_xticks(range(NC), [c[:7] for c in CL], fontsize=6.5)
        a.set_yticks(range(NC), [c[:7] for c in CL], fontsize=6.5)
        a.grid(False)
        a.set_title(label(m), fontsize=7)
    fig.supxlabel("predicted", fontsize=8); fig.supylabel("true", fontsize=8)
    fig.tight_layout()
    save(fig, FIG / "confusion_matrices")


def subject_heatmap():
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.3 * len(MODELS)))
    im = ax.imshow(subj_acc.T.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(MODELS)), [label(m) for m in MODELS], fontsize=6.5)
    ax.set_xticks(range(len(subj_acc)), [s[:4] for s in subj_acc.index], fontsize=6, rotation=90)
    ax.grid(False)
    for i in range(len(MODELS)):
        for j in range(len(subj_acc)):
            v = subj_acc.iloc[j, i]
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, label="window accuracy")
    ax.set_title(f"{args.task} / {args.protocol}: accuracy per held-out subject (LOSO, mean over seeds)", fontsize=8)
    save(fig, FIG / "subject_heatmap")


def robustness():
    rows_ = []
    for m in fus:
        mods = modalities(m, cfg)
        for seed, rs in RUN[m].items():
            P, y, *_ = pooled(rs)
            rows_.append(dict(model=m, seed=seed, input="all", f1=M.classification(P, y, n_cls=NC)["f1_macro"]))
            for mod in mods:
                P, y, *_ = pooled(rs, f"prob_no_{mod}")
                rows_.append(dict(model=m, seed=seed, input=f"without {mod}", f1=M.classification(P, y, n_cls=NC)["f1_macro"]))
    df = pd.DataFrame(rows_)
    if df.empty:
        return
    t = df.groupby(["model", "input"], sort=False).f1.mean().unstack()
    t.to_csv(OUT / "missing_modality.csv")
    print("\nmissing-modality macro-F1\n", t.round(3).to_string())
    labs = ["all"] + [f"without {m}" for m in cfg["modalities"]]
    cols = ["#256abf", "#eb6834", "#1baf7a", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(fus)), 3))
    w = 0.8 / len(labs)
    for k, (lab, c) in enumerate(zip(labs, cols)):
        g = df[df.input == lab].groupby("model", sort=False).f1
        mu, sd = g.mean().reindex(fus), g.std().reindex(fus)
        ax.bar(np.arange(len(fus)) + (k - (len(labs) - 1) / 2) * w, mu, w * 0.92, yerr=sd, color=c, label=lab,
               error_kw=dict(lw=0.7, ecolor=INK2), zorder=2)
    ax.axhline(1 / NC, c=GRAY, ls="--", lw=0.8)
    ax.set_xticks(range(len(fus)), [short(m) for m in fus], fontsize=7.5)
    ax.set_ylabel("macro-F1")
    ax.legend(fontsize=7, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, FIG / "missing_modality")


def fusion_weights():
    out = {}
    panels = [(m, k) for m in fus for k in ("x_mod_weights", "x_level_weights")
              if k in RUN[m][min(RUN[m])][min(RUN[m][min(RUN[m])])]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 2.6))
    for a, (m, k) in zip(np.atleast_1d(axes), panels):
        W = np.concatenate([rs[s][k] for rs in RUN[m].values() for s in rs])
        y = np.concatenate([rs[s]["y"] for rs in RUN[m].values() for s in rs])
        names = [x.upper() for x in modalities(m, cfg)] if "mod" in k else [f"L{i + 1}" for i in range(W.shape[1])]
        x = np.arange(W.shape[1])
        for ci_, c in enumerate(CL):
            a.bar(x + (ci_ - (NC - 1) / 2) * 0.8 / NC, W[y == ci_].mean(0), 0.75 / NC, color=CCOL[ci_], label=c, zorder=2)
        a.set_xticks(x, names)
        a.set_title(f"{short(m)}: {'modality' if 'mod' in k else 'level'} weights", fontsize=8)
        a.grid(axis="x", visible=False)
        out[m] = {n: float(v) for n, v in zip(names, W.mean(0))}
    np.atleast_1d(axes)[0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, FIG / "fusion_weights")
    (OUT / "fusion_weights.json").write_text(json.dumps(out, indent=1))


def learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for m in MODELS:
        H = [rs[s]["hist"] for rs in RUN[m].values() for s in rs]
        n = max(len(h) for h in H)
        for a, key in zip(axes, ("train_loss", "val_f1")):
            arr = np.full((len(H), n), np.nan)
            for i, h in enumerate(H):
                arr[i, :len(h)] = h[key].values
            a.plot(np.nanmean(arr, 0), c=FAMILY_COLORS[family(m)], lw=1.1, ls="--" if family(m) == "Unimodal" else "-",
                   label=short(m))
    axes[0].set_ylabel("train loss"); axes[1].set_ylabel("validation macro-F1")
    for a in axes:
        a.set_xlabel("epoch")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, FIG / "learning_curves")


def embeddings():
    rs = RUN[best][min(RUN[best])]
    Z = np.concatenate([rs[s]["emb"] for s in sorted(rs)]).astype(np.float32)
    y = np.concatenate([rs[s]["y"] for s in sorted(rs)])
    sid = np.concatenate([rs[s]["sid"] for s in sorted(rs)])
    T = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    for k, c in enumerate(CL):
        axes[0].scatter(T[y == k, 0], T[y == k, 1], s=3, color=CCOL[k], label=c)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(np.unique(sid)):
        axes[1].scatter(T[sid == s, 0], T[sid == s, 1], s=3, color=cmap(i % 20))
    axes[0].legend(markerscale=3, fontsize=7)
    axes[0].set_title(f"{label(best)}: held-out embeddings by condition", fontsize=7.5)
    axes[1].set_title("by subject", fontsize=7.5)
    for a in axes:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, FIG / "embeddings_tsne")


@torch.no_grad()
def efficiency():
    rows_ = []
    L = cfg["window"]["length_s"]
    x = {m: torch.randn(64, st["ch"], L * st["fs"], device="cuda") for m, st in cfg["streams"].items()}
    for m in MODELS:
        net = build(m, cfg).cuda().eval()
        for _ in range(5):
            net(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            net(x)
        torch.cuda.synchronize()
        info = [rs[s]["info"] for rs in RUN[m].values() for s in rs]
        rows_.append(dict(model=m, params_M=sum(p.numel() for p in net.parameters()) / 1e6,
                          latency_ms_per_window=(time.time() - t0) / 20 / 64 * 1000,
                          train_time_s=np.mean([i["train_time_s"] for i in info]),
                          best_epoch=np.mean([i["best_epoch"] for i in info])))
    pd.DataFrame(rows_).to_csv(OUT / "efficiency.csv", index=False)


bar_f1()
cd_diagram()
confusion_grid(MODELS)
subject_heatmap()
robustness()
fusion_weights()
learning_curves()
embeddings()
efficiency()
print("\nbest model (subject-level F1):", best)
