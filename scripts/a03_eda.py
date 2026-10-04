"""Exploratory analysis: cohort, data quality, manipulation check (NASA-TLX,
task performance per level) and physiological responses per level relative to
the rest baseline, with repeated-measures statistics (subject = block)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg, seg_key, subjects
from fada import ecg as E
from fada import stats as S
from fada.plotting import GRAY, INK2, SLOTS, save, style

cfg = load_cfg()
DATA = Path(cfg["paths"]["data"])
OUT = Path(cfg["paths"]["results"]) / "eda"
FIG = Path(cfg["paths"]["figures"]) / "eda"
LEVELS = {"n-back": [("baseline", 1, "Rest"), ("baseline", 2, "Stimulus only"), ("test", 1, "1-back"),
                     ("test", 2, "2-back"), ("test", 3, "3-back")],
          "k-drive": [("baseline", 1, "Rest"), ("test", 1, "Drive L1"), ("test", 2, "Drive L2"), ("test", 3, "Drive L3")]}
COLORS = ["#8a8985", "#9ec5f4", SLOTS[0], SLOTS[1], SLOTS[2]]


def paired(ax, w, ylabel, cols):
    x = np.arange(len(cols))
    for _, r in w.iterrows():
        ax.plot(x, r[cols].values, c="#d4d3cf", lw=0.7, zorder=1)
    for i, c in enumerate(cols):
        ax.boxplot(w[c].dropna(), positions=[i], widths=0.45, showfliers=False,
                   boxprops=dict(color=INK2), medianprops=dict(color=INK2, lw=1.2), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2))
        ax.scatter(np.full(len(w), i), w[c], s=12, c=COLORS[-len(cols):][i] if len(cols) <= 5 else SLOTS[0], zorder=3,
                   edgecolor="white", lw=0.4)
    ax.set_xticks(x, cols, rotation=20, fontsize=7.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def segment_features():
    rows = []
    for sub in subjects(cfg):
        z = np.load(DATA / "streams" / f"{sub}.npz")
        for study, lv in LEVELS.items():
            for phase, level, name in lv:
                k = seg_key(study, phase, level)
                if f"{k}|ecg" not in z.files:
                    continue
                pk = z[f"{k}|rpeaks"] if f"{k}|rpeaks" in z.files else np.array([])
                h = E.hrv_features(pk, 500) if len(pk) > 10 else dict(hr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
                rsp, emg = z[f"{k}|rsp"], z[f"{k}|emg"]
                rows.append(dict(subject=sub, study=study, level=name, hr=h["hr"], rmssd=h["rmssd"], sdnn=h["sdnn"],
                                 br=float(np.median(rsp[1] * 5 + 15)), emg_low=float(np.mean(emg[0])),
                                 emg_mid=float(np.mean(emg[1])), emg_high=float(np.mean(emg[2]))))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    # ------------------------------------------------------------ cohort
    meta = pd.read_csv(DATA / "meta.csv")
    age = [c for c in meta.columns if c.endswith("Age")][0]
    sex = [c for c in meta.columns if c.endswith("Sex")][0]
    q = pd.read_csv(DATA / "quality.csv")
    seg = pd.read_csv(DATA / "segments.csv")
    import yaml
    excl = yaml.safe_load((DATA / "exclusions.yaml").read_text())
    summ = [("Subjects (files)", str(len(meta))), ("Age (years), mean ± SD [range]",
            f"{meta[age].mean():.1f} ± {meta[age].std():.1f} [{meta[age].min():.0f}, {meta[age].max():.0f}]"),
            ("Sex", ", ".join(f"{k}: {v}" for k, v in meta[sex].value_counts().items())),
            ("Signals used", "ECG (2 leads), EMG (trapezius), respiration (chest belt)"),
            ("Stored sampling", "common 1000 Hz timeline (native ECG 500 Hz, RSP 250 Hz, EMG 1000 Hz)"),
            ("Segments per subject", f"{seg.groupby('subject').size().median():.0f}"),
            ("ECG lead I / II pass SQI", f"{q.lead1_ok.mean():.3f} / {q.lead2_ok.mean():.3f} of segments"),
            ("Excluded (n-back tasks)", ", ".join(excl.get("nback_bin", [])) or "none"),
            ("Excluded (k-drive tasks)", ", ".join(excl.get("kdrive_3", [])) or "none")]
    pd.DataFrame(summ, columns=["item", "value"]).to_csv(OUT / "dataset_summary.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False, header=False))

    # ------------------------------------------------- manipulation check
    sj = pd.read_csv(DATA / "subjective.csv")
    w = sj[[c for c in sj.columns if c.startswith("WEIGHT")]].values
    r = sj[["EFFORT", "FRUSTRATION", "MENTAL", "PERFORMANCE", "PHYSICAL", "TEMPORAL"]].values
    sj["TLX"] = (w * r).sum(1) / np.maximum(w.sum(1), 1)
    pf = pd.read_csv(DATA / "performance.csv")
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for a, (study, levels, var, src, lab) in zip(axes, [
            ("n-back", [1, 2, 3], "TLX", sj, "NASA-TLX (weighted)"), ("k-drive", [1, 2, 3], "TLX", sj, "NASA-TLX (weighted)"),
            ("n-back", [1, 2, 3], "VISUAL F1", pf, "n-back visual F1"), ("k-drive", [1, 2, 3], "F1", pf, "k-drive task F1")]):
        d = src[(src.STUDY == study) & (src.PHASE == "test") & (src.LEVEL.isin(levels))]
        wd = d.pivot_table(index="subject", columns="LEVEL", values=var).dropna()
        wd.columns = [f"L{c}" for c in wd.columns]
        fr = S.friedman(wd.values)
        rows.append(dict(study=study, measure=var, **{c: wd[c].mean() for c in wd.columns}, **fr))
        paired(a, wd.reset_index(), lab, list(wd.columns))
        a.set_title(f"{study}: Friedman p={fr['p']:.2g}, W={fr['kendall_w']:.2f}", fontsize=7.5)
    fig.tight_layout()
    save(fig, FIG / "manipulation_check")
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "manipulation_check.csv", index=False)
    print(mc.round(4).to_string())

    # ------------------------------------------- physiology per level
    f = segment_features()
    f.to_csv(OUT / "segment_features.csv", index=False)
    feats = {"hr": "HR (bpm)", "rmssd": "RMSSD (ms)", "br": "Breathing rate (/min)", "emg_high": "EMG log-RMS 150-450 Hz",
             "emg_low": "EMG log-RMS 20-60 Hz"}
    rows, post = [], []
    for study, lv in LEVELS.items():
        names = [n for _, _, n in lv]
        fig, axes = plt.subplots(1, len(feats), figsize=(13, 3))
        for a, (k, lab) in zip(axes, feats.items()):
            wd = f[f.study == study].pivot(index="subject", columns="level", values=k)[names].dropna()
            test_cols = [n for n in names if n != "Rest"]
            fr = S.friedman(wd[test_cols].values)
            pw = S.pairwise_wilcoxon(wd[test_cols])
            pw.insert(0, "feature", k); pw.insert(0, "study", study)
            post.append(pw)
            rows.append(dict(study=study, feature=k, **{nm: wd[nm].mean() for nm in names}, **fr))
            paired(a, wd.reset_index(), lab, names)
            a.set_title(f"task levels: p={fr['p']:.2g}, W={fr['kendall_w']:.2f}", fontsize=7.5)
        fig.suptitle(f"{study}: physiology per level (Friedman across task levels, rest shown for reference)")
        fig.tight_layout()
        save(fig, FIG / f"physiology_{study.replace('-', '')}")
    ph = pd.DataFrame(rows)
    ph["p_holm"] = multipletests(ph.p, method="holm")[1]
    ph.to_csv(OUT / "physiology_friedman.csv", index=False)
    pd.concat(post).to_csv(OUT / "physiology_posthoc.csv", index=False)
    print(ph.round(4).to_string())

    # ------------------------------------------------- example traces
    sub = subjects(cfg, "nback_bin")[0]
    z = np.load(DATA / "streams" / f"{sub}.npz")
    fig, axes = plt.subplots(3, 3, figsize=(11, 5.5), sharex="col")
    for j, (lvl, name) in enumerate([(("baseline", 1), "Rest"), (("test", 1), "1-back"), (("test", 3), "3-back")]):
        k = seg_key("n-back", *lvl)
        e, m, r = z[f"{k}|ecg"], z[f"{k}|emg"], z[f"{k}|rsp"]
        t0 = 60
        axes[0, j].plot(np.arange(5 * 125) / 125, e[1, t0 * 125:(t0 + 5) * 125], c=SLOTS[0], lw=0.8)
        for b, c in zip(range(3), (SLOTS[0], SLOTS[1], SLOTS[2])):
            axes[1, j].plot(np.arange(20 * 50) / 50, m[b, t0 * 50:(t0 + 20) * 50], c=c, lw=0.8, label=["20-60", "60-150", "150-450 Hz"][b])
        axes[2, j].plot(np.arange(20 * 25) / 25, r[0, t0 * 25:(t0 + 20) * 25], c=SLOTS[2], lw=1)
        axes[0, j].set_title(f"{name}", fontsize=9)
    axes[0, 0].set_ylabel("ECG lead II\n(5 s)"); axes[1, 0].set_ylabel("EMG log-RMS\n(20 s)"); axes[2, 0].set_ylabel("Respiration\n(20 s)")
    axes[1, 2].legend(fontsize=6.5)
    for a in axes[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle(f"Pre-processed streams, subject {sub} (n-back)")
    fig.tight_layout()
    save(fig, FIG / "stream_examples")


if __name__ == "__main__":
    main()
