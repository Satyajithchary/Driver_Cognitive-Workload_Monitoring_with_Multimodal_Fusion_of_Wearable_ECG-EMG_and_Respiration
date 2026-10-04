"""Ablations on the main task (n-back binary, enroll protocol, seed 0):
  streams   each fusion over every stream pair vs all three streams
  win10/30  10-s / 30-s windows instead of 20 s
  python scripts/a08_ablations.py --run   |   --summarise"""
import argparse
import subprocess
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
from fada.plotting import FAMILY_COLORS, GRAY, INK2, family, save, short, style

cfg = load_cfg()
ROOT = Path(__file__).resolve().parents[1]
RES = Path(cfg["paths"]["results"])
ABL = RES / "ablations"
TASK = cfg["main_task"]
MAIN = RES / "runs" / TASK / "enroll"
PAIRS = ["ecg+emg", "ecg+rsp", "emg+rsp"]
FUS = ["inter_concat", "late", "hot"]
NC = len(cfg["tasks"][TASK]["classes"])
GROUPS = {
    "streams": dict(models=[f"{f}@{p}" for f in FUS for p in PAIRS], sets=[], shards=5),
    "win10": dict(models=cfg["unimodal"] + ["inter_concat", "late"], sets=["window.length_s=10"], shards=2),
    "win30": dict(models=cfg["unimodal"] + ["inter_concat", "late"], sets=["window.length_s=30"], shards=2),
}


def run():
    procs = []
    for g, spec in GROUPS.items():
        for i in range(spec["shards"]):
            cmd = [sys.executable, "-u", "scripts/a06_train.py", "--task", TASK, "--protocol", "enroll", "--seeds", "0",
                   "--out", str(ABL / g), "--models", *spec["models"], "--shard", str(i), "--nshards", str(spec["shards"])]
            if spec["sets"]:
                cmd += ["--set", *spec["sets"]]
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=open(ROOT / f"logs/abl_{g}_{i}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


def load(d, model):
    out = {}
    for f in sorted((d / "preds" / model).glob("*_seed0.npz")):
        out[f.stem.split("_seed")[0]] = np.load(f)
    return out


def score(d, model):
    rs = load(d, model)
    if not rs:
        return None, None
    P = np.concatenate([rs[s]["prob"] for s in rs])
    y = np.concatenate([rs[s]["y"] for s in rs])
    subj = pd.Series({s: classification(rs[s]["prob"], rs[s]["y"], n_cls=NC)["f1_macro"] for s in rs})
    return classification(P, y, n_cls=NC), subj


def summarise():
    style()
    rows, subj = [], {}
    for m in cfg["unimodal"] + FUS:
        res, sf = score(MAIN, m)
        if res:
            rows.append(dict(group="main (20 s, 3 streams)", model=m, f1=res["f1_macro"], acc=res["acc"], auroc=res["auroc"]))
            subj[("main", m)] = sf
    for g, spec in GROUPS.items():
        for m in spec["models"]:
            res, sf = score(ABL / g / TASK / "enroll", m)
            if res:
                rows.append(dict(group=g, model=m, f1=res["f1_macro"], acc=res["acc"], auroc=res["auroc"]))
                subj[(g, m)] = sf
    df = pd.DataFrame(rows)
    df.to_csv(ABL / "ablation_summary.csv", index=False)
    print(df.round(3).to_string())
    st = []
    for f in FUS:
        for p in PAIRS:
            if ("main", f) in subj and ("streams", f"{f}@{p}") in subj:
                a, b = subj[("main", f)], subj[("streams", f"{f}@{p}")].reindex(subj[("main", f)].index)
                pw = S.pairwise_wilcoxon(pd.DataFrame({"all3": a, p: b}).dropna())
                st.append(dict(fusion=f, pair=p, f1_all3=a.mean(), f1_pair=b.mean(), p=pw.p.iloc[0], r_rb=pw.r_rb.iloc[0]))
    if st:
        st = pd.DataFrame(st)
        st["p_holm"] = multipletests(st.p, method="holm")[1]
        st.to_csv(ABL / "streams_wilcoxon.csv", index=False)
        print(st.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    a = axes[0]
    labels, vals, cols = [], [], []
    for m in cfg["unimodal"]:
        r = df[(df.group.str.startswith("main")) & (df.model == m)]
        labels.append(short(m)); vals.append(r.f1.iloc[0] if len(r) else np.nan); cols.append(GRAY)
    for f in FUS:
        for p in PAIRS + ["all"]:
            name = f if p == "all" else f"{f}@{p}"
            r = df[((df.group.str.startswith("main")) if p == "all" else (df.group == "streams")) & (df.model == name)]
            labels.append(f"{short(f)}(E+M+R)" if p == "all" else short(name))
            vals.append(r.f1.iloc[0] if len(r) else np.nan)
            cols.append(FAMILY_COLORS[family(f)] if p == "all" else "#c8d9f0")
    a.bar(range(len(vals)), vals, 0.7, color=cols, zorder=2)
    a.axhline(1 / NC, c=GRAY, ls="--", lw=0.8)
    a.set_xticks(range(len(vals)), labels, rotation=50, ha="right", fontsize=6.5)
    a.set_ylabel("window macro-F1 (seed 0)")
    a.set_title("stream subsets", fontsize=8)
    a.grid(axis="x", visible=False)
    for m in cfg["unimodal"] + ["inter_concat", "late"]:
        pts = []
        for grp in ("win10", "main (20 s, 3 streams)", "win30"):
            r = df[(df.group == grp) & (df.model == m)]
            pts.append(r.f1.iloc[0] if len(r) else np.nan)
        axes[1].plot([10, 20, 30], pts, marker="o", ms=4, lw=1.4, ls="--" if m.endswith("_only") else "-",
                     color=FAMILY_COLORS[family(m)] if not m.endswith("_only") else {"ecg_only": "#0b0b0b", "emg_only": "#52514e", "rsp_only": GRAY}[m],
                     label=short(m))
    axes[1].set_xticks([10, 20, 30]); axes[1].set_xlabel("window length (s)"); axes[1].set_ylabel("window macro-F1 (seed 0)")
    axes[1].legend(fontsize=7); axes[1].set_title("window length", fontsize=8)
    fig.tight_layout()
    save(fig, Path(cfg["paths"]["figures"]) / "ablations" / "ablations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.summarise:
        summarise()
