"""Hand-crafted feature baselines (same LOSO windows / protocols as the deep
models) and a within-subject label-permutation test for every task (1000 permutations
for the main task / main protocol, 200 elsewhere).

Window features (20 s): ECG mean/SD/slope of iHR, SDNN, RMSSD; RSP mean/SD of
breathing rate, waveform amplitude (SD of the robust-scaled signal); EMG mean/SD
of the three log-RMS bands.
Calibration: strict none; enroll = subtract the mean of the subject's rest-baseline
windows (baseline correction, as in the deep models); calib = z-score with all of
the subject's task windows (no labels)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg, seg_key, subjects
from fada import ecg as E
from fada.metrics import classification

cfg = load_cfg()
DATA = Path(cfg["paths"]["data"])
OUT = Path(cfg["paths"]["results"]) / "classical"
L, STRIDE = cfg["window"]["length_s"], cfg["window"]["stride_eval_s"]
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
    "rf": lambda: RandomForestClassifier(300, min_samples_leaf=5, n_jobs=8, random_state=0),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
}


def window_features(task):
    t = cfg["tasks"][task]
    rows = []
    for sub in subjects(cfg, task):
        z = np.load(DATA / "streams" / f"{sub}.npz")
        items = [(y, s, False, si) for y, cls in enumerate(t["segments"]) for si, s in enumerate(cls)] + [(-1, t["reference"], True, 0)]
        for y, s, ref, si in items:
            k = seg_key(*s)
            e, m, r = z[f"{k}|ecg"], z[f"{k}|emg"], z[f"{k}|rsp"]
            pk = z[f"{k}|rpeaks"] if f"{k}|rpeaks" in z.files else np.array([])
            dur = min(e.shape[1] / 125, m.shape[1] / 50, r.shape[1] / 25)
            for st in np.arange(0, dur - L + 1e-9, STRIDE):
                hr = e[2, int(st * 125):int((st + L) * 125)] * 15 + 80
                hv = E.hrv_features(pk, 500, st, st + L) if len(pk) else dict(sdnn=np.nan, rmssd=np.nan)
                br = r[1, int(st * 25):int((st + L) * 25)] * 5 + 15
                f = dict(subject=sub, y=y, ref=ref, seg=f"{k}", start=st,
                         ecg_hr=hr.mean(), ecg_hr_sd=hr.std(), ecg_hr_slope=np.polyfit(np.arange(len(hr)), hr, 1)[0] * 125,
                         ecg_sdnn=hv["sdnn"], ecg_rmssd=hv["rmssd"],
                         rsp_rate=br.mean(), rsp_rate_sd=br.std(), rsp_amp=r[0, int(st * 25):int((st + L) * 25)].std())
                for b in range(3):
                    v = m[b, int(st * 50):int((st + L) * 50)]
                    f[f"emg_b{b}"] = v.mean()
                    f[f"emg_b{b}_sd"] = v.std()
                rows.append(f)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median())
    return df


def calibrate(df, cols, protocol):
    X = df[cols].values.astype(np.float64).copy()
    if protocol == "strict":
        return X
    for s in df.subject.unique():
        m = (df.subject == s).values
        if protocol == "enroll":
            X[m] = X[m] - X[m & df.ref.values].mean(0)
        else:
            t = m & ~df.ref.values
            X[m] = (X[m] - X[t].mean(0)) / (X[t].std(0) + 1e-6)
    return X


def loso(X, y, groups, train, test, make):
    P = np.zeros((len(y), int(y[train].max()) + 1))
    for s in np.unique(groups[test]):
        tr, te = train & (groups != s), test & (groups == s)
        P[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])
    return P


def permute(df, rng):
    """Shuffle the class labels across each subject's task segments (segment
    level, keeping class sizes), i.e. labels lose meaning but structure stays."""
    y = df.y.values.copy()
    for s in df.subject.unique():
        m = (df.subject == s).values & ~df.ref.values
        segs = df.loc[m, "seg"].unique()
        lab = {sg: df.loc[m & (df.seg == sg).values, "y"].iloc[0] for sg in segs}
        perm = dict(zip(segs, rng.permutation([lab[sg] for sg in segs])))
        y[m] = df.loc[m, "seg"].map(perm).values
    return y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, perm = [], {}
    for task in cfg["tasks"]:
        df = window_features(task)
        df.to_csv(OUT / f"window_features_{task}.csv", index=False)
        groups = {"ecg": [c for c in df if c.startswith("ecg_")], "emg": [c for c in df if c.startswith("emg_")],
                  "rsp": [c for c in df if c.startswith("rsp_")]}
        groups["all"] = groups["ecg"] + groups["emg"] + groups["rsp"]
        y, g, isref = df.y.values, df.subject.values, df.ref.values
        task_rows = ~isref
        n_cls = len(cfg["tasks"][task]["classes"])
        for protocol in cfg["protocols"]:
            for fset, cols in groups.items():
                X = calibrate(df, cols, protocol)
                for mname, make in MODELS.items():
                    P = loso(X, y, g, task_rows, task_rows, make)
                    res = classification(P[task_rows], y[task_rows], n_cls=n_cls)
                    sess = df[task_rows].assign(**{f"p{k}": P[task_rows, k] for k in range(n_cls)}) \
                        .groupby(["subject", "seg"])[[f"p{k}" for k in range(n_cls)] + ["y"]].mean()
                    rs = classification(sess[[f"p{k}" for k in range(n_cls)]].values, sess.y.values.astype(int), n_cls=n_cls)
                    rows.append(dict(task=task, protocol=protocol, features=fset, model=mname, acc=res["acc"],
                                     f1_macro=res["f1_macro"], auroc=res["auroc"], seg_acc=rs["acc"], seg_f1=rs["f1_macro"]))
                    print(f"{task:10s} {protocol:6s} {fset:4s} {mname:6s} acc={res['acc']:.3f} f1={res['f1_macro']:.3f} "
                          f"seg_acc={rs['acc']:.3f}", flush=True)
        # permutation test, all features + logistic regression, every protocol
        perm[task] = {}
        for protocol in cfg["protocols"]:
            X = calibrate(df, groups["all"], protocol)
            obs = [r for r in rows if r["task"] == task and r["protocol"] == protocol and r["features"] == "all"
                   and r["model"] == "logreg"][0]["f1_macro"]
            rng = np.random.RandomState(0)
            null = []
            n_perm = 1000 if (task == cfg["main_task"] and protocol == "enroll") else 200
            for _ in range(n_perm):
                yp = permute(df, rng)
                P = loso(X, yp, g, task_rows, task_rows, MODELS["logreg"])
                null.append(classification(P[task_rows], yp[task_rows], n_cls=n_cls)["f1_macro"])
            null = np.array(null)
            perm[task][protocol] = dict(observed_f1=obs, null_mean=null.mean(), null_95=float(np.percentile(null, 95)),
                                        p=float((1 + (null >= obs).sum()) / (1 + n_perm)), n_perm=n_perm)
            print("permutation", task, protocol, perm[task][protocol], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "classical_results.csv", index=False)
    (OUT / "permutation_test.json").write_text(json.dumps(perm, indent=1))


if __name__ == "__main__":
    main()
