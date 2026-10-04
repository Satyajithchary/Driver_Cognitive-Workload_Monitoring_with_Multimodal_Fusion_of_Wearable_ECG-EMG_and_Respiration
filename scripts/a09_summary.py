"""Cross-task / cross-protocol summary: paper tables (markdown + CSV), protocol
effect tests on the main task, classical and permutation results, figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg
from fada import stats as S
from fada.metrics import classification
from fada.plotting import GRAY, INK2, SEQ_CMAP, label, save, short, style

cfg = load_cfg()
RES = Path(cfg["paths"]["results"])
OUT = RES / "summary"
FIG = Path(cfg["paths"]["figures"]) / "summary"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = cfg["unimodal"] + cfg["fusions"]
MAIN = cfg["main_task"]
style()


def load(task, protocol, model, seeds=None):
    out = {}
    for f in sorted((RES / "runs" / task / protocol / "preds" / model).glob("*_seed*.npz")):
        sub, seed = f.stem.split("_seed")
        if seeds is None or int(seed) in seeds:
            out.setdefault(int(seed), {})[sub] = np.load(f)
    return out


def scores(task, protocol, model, seeds=None):
    nc = len(cfg["tasks"][task]["classes"])
    rows = []
    for seed, rs in load(task, protocol, model, seeds).items():
        P = np.concatenate([rs[s]["prob"] for s in rs]); y = np.concatenate([rs[s]["y"] for s in rs])
        rec = np.concatenate([np.char.add(f"{s}|", rs[s]["rec"].astype(str)) for s in rs])
        w = classification(P, y, n_cls=nc)
        u = np.unique(rec)
        Ps = np.stack([P[rec == k].mean(0) for k in u]); ys = np.array([y[rec == k][0] for k in u])
        sg = classification(Ps, ys, n_cls=nc)
        rows.append(dict(task=task, protocol=protocol, model=model, seed=seed, n_subj=len(rs), f1=w["f1_macro"],
                         acc=w["acc"], auroc=w["auroc"], kappa=w["kappa"], seg_acc=sg["acc"], seg_f1=sg["f1_macro"]))
    return rows


def subject_f1(task, protocol, model, seeds):
    nc = len(cfg["tasks"][task]["classes"])
    per = [{s: classification(rs[s]["prob"], rs[s]["y"], n_cls=nc)["f1_macro"] for s in rs}
           for rs in load(task, protocol, model, seeds).values()]
    return pd.DataFrame(per).mean() if per else pd.Series(dtype=float)


def fmt(m, s):
    return f"{m:.3f}" if np.isnan(s) else f"{m:.3f} ± {s:.3f}"


rows = []
for task in cfg["tasks"]:
    for protocol in cfg["protocols"]:
        for m in MODELS:
            rows += scores(task, protocol, m)
sc = pd.DataFrame(rows)
sc.to_csv(OUT / "all_scores_per_seed.csv", index=False)
agg = sc.groupby(["task", "protocol", "model"])[["f1", "acc", "auroc", "kappa", "seg_acc", "seg_f1"]].agg(["mean", "std"])

lines = ["# ADABase results summary (LOSO; main task enroll: mean ± SD over 3 seeds; otherwise seed 0)", ""]
for task in cfg["tasks"]:
    for protocol in cfg["protocols"]:
        if (task, protocol) not in {(a, b) for a, b, _ in agg.index}:
            continue
        nc = len(cfg["tasks"][task]["classes"])
        lines += [f"## {task} — `{protocol}` (chance {1 / nc:.2f})", "",
                  "| Model | Window F1 | Window Acc | AUROC | κ | Segment Acc | Segment F1 |", "|---|---|---|---|---|---|---|"]
        for m in MODELS:
            if (task, protocol, m) in agg.index:
                r = agg.loc[(task, protocol, m)]
                lines.append(f"| {label(m)} | " + " | ".join(fmt(r[(k, 'mean')], r[(k, 'std')]) for k in
                                                             ("f1", "acc", "auroc", "kappa", "seg_acc", "seg_f1")) + " |")
        lines.append("")

# protocol effect (main task, seed 0 in every protocol)
pe = []
for m in MODELS:
    f = {p: subject_f1(MAIN, p, m, [0]) for p in cfg["protocols"]}
    if any(len(v) == 0 for v in f.values()):
        continue
    for a, b in (("enroll", "strict"), ("calib", "enroll")):
        d = pd.DataFrame({a: f[a], b: f[b]}).dropna()
        pw = S.pairwise_wilcoxon(d)
        pe.append(dict(model=m, comparison=f"{a} vs {b}", mean_a=d[a].mean(), mean_b=d[b].mean(), p=pw.p.iloc[0], r_rb=pw.r_rb.iloc[0]))
pe = pd.DataFrame(pe)
if len(pe):
    pe["p_holm"] = multipletests(pe.p, method="holm")[1]
    pe.to_csv(OUT / "protocol_effect.csv", index=False)
    lines += [f"## Protocol effect on {MAIN} (subject-level macro-F1, seed 0, Wilcoxon, Holm)", "", pe.round(4).to_markdown(index=False), ""]

cl = RES / "classical" / "classical_results.csv"
if cl.exists():
    c = pd.read_csv(cl)
    best = c.loc[c.groupby(["task", "protocol", "features"]).f1_macro.idxmax()]
    lines += ["## Classical feature baselines (best of LR / RF / HGB)", "",
              best[["task", "protocol", "features", "model", "acc", "f1_macro", "auroc", "seg_acc"]].round(3).to_markdown(index=False), ""]
pt = RES / "classical" / "permutation_test.json"
if pt.exists():
    p = json.loads(pt.read_text())
    tab = pd.DataFrame([dict(task=t, protocol=k, **v) for t, d in p.items() for k, v in d.items()])
    lines += ["## Permutation test (LR, all features, 1000 within-subject label permutations)", "", tab.round(4).to_markdown(index=False), ""]
ab = RES / "ablations" / "ablation_summary.csv"
if ab.exists():
    lines += [f"## Ablations ({MAIN}, enroll, seed 0)", "", pd.read_csv(ab).round(3).to_markdown(index=False), ""]
sw = RES / "ablations" / "streams_wilcoxon.csv"
if sw.exists():
    lines += ["### Three streams vs each pair", "", pd.read_csv(sw).round(4).to_markdown(index=False), ""]
(OUT / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))

# figures --------------------------------------------------------------------
m_ = agg.loc[MAIN] if MAIN in agg.index.get_level_values(0) else None
if m_ is not None:
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    x = np.arange(len(MODELS))
    cols = {"strict": "#cde2fb", "enroll": "#6da7ec", "calib": "#184f95"}
    for k, p in enumerate(cfg["protocols"]):
        mu = [m_.loc[(p, m)][("f1", "mean")] if (p, m) in m_.index else np.nan for m in MODELS]
        sd = [m_.loc[(p, m)][("f1", "std")] if (p, m) in m_.index else 0 for m in MODELS]
        ax.bar(x + (k - 1) * 0.27, mu, 0.25, yerr=np.nan_to_num(sd), color=cols[p], label=p, error_kw=dict(lw=0.7, ecolor=INK2), zorder=2)
    ax.axhline(0.5, c=GRAY, ls="--", lw=0.8)
    ax.set_xticks(x, [short(m) for m in MODELS], fontsize=7.5)
    ax.set_ylabel("macro-F1 (20-s window)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{MAIN}: protocols", fontsize=8, loc="right")
    save(fig, FIG / "protocols_main_task")

ent = sc[sc.protocol == "enroll"].groupby(["task", "model"]).f1.mean().unstack().reindex(columns=MODELS)
if len(ent):
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.45 * len(ent)))
    im = ax.imshow(ent.values, cmap=SEQ_CMAP, vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_yticks(range(len(ent)), ent.index)
    ax.set_xticks(range(len(MODELS)), [short(m) for m in MODELS], fontsize=7.5)
    ax.grid(False)
    for i in range(len(ent)):
        for j in range(len(MODELS)):
            v = ent.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.65 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="window macro-F1")
    ax.set_title("enroll protocol: window macro-F1 per task and model (chance 0.50 binary / 0.33 three-class)", fontsize=8)
    save(fig, FIG / "tasks_models_heatmap")
