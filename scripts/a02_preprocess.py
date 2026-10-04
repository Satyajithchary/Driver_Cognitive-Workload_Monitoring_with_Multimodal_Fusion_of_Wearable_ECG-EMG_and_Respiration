"""Pre-processing and quality audit of every segment used by any task.

ECG  (2 leads, 500 Hz): 0.5-40 Hz band-pass, polarity check per lead, R-peaks
     and SQI per lead, HR from the better lead -> [lead I, lead II, iHR] at 125 Hz
EMG  (1000 Hz): 20-450 Hz band-pass, 50 Hz notches -> log-RMS envelopes of
     20-60 / 60-150 / 150-450 Hz at 50 Hz
RSP  (250 Hz): 0.05-1 Hz band-pass, breath peaks -> [waveform, breathing rate] at 25 Hz
A subject is excluded from a task when a needed segment is missing / too short
or a stream fails its quality check in any needed segment."""
import sys
from multiprocessing import Pool
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import yaml
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg, seg_key
from fada import ecg as E
from fada import signals as SG

cfg = load_cfg()
DATA = Path(cfg["paths"]["data"])
OUT = DATA / "streams"
HR_MU, HR_SD, BR_MU, BR_SD = 80.0, 15.0, 15.0, 5.0


def needed_segments():
    segs = set()
    for t in cfg["tasks"].values():
        segs.add(tuple(t["reference"]))
        for cls in t["segments"]:
            for s in cls:
                segs.add(tuple(s))
    return sorted(segs)


def robust_z(x):
    med = np.median(x)
    return np.clip((x - med) / (1.4826 * np.median(np.abs(x - med)) + 1e-9), -8, 8)


def process(sub):
    z = np.load(DATA / "raw_segments" / f"{sub}.npz")
    out, rows = {}, []
    for study, phase, level in needed_segments():
        key = seg_key(study, phase, level)
        row = dict(subject=sub, study=study, phase=phase, level=level, present=f"{key}|ecg" in z.files)
        if not row["present"]:
            rows.append(row)
            continue
        ecg2, rsp, emg = z[f"{key}|ecg"].astype(np.float64), z[f"{key}|rsp"].astype(np.float64), z[f"{key}|emg"].astype(np.float64)
        dur = len(emg) / 1000
        leads, best = [], None
        for li in range(2):
            y = E.bandpass(ecg2[:, li], 500, (0.5, 40))
            y, inv = nk.ecg_invert(y, sampling_rate=500)
            pk = E.r_peaks(y, 500)
            q = E.sqi(y, 500, pk)
            row.update({f"lead{li + 1}_{k}": q.get(k) for k in ("hr", "tcorr", "coverage", "rr_cv", "ok")})
            row[f"lead{li + 1}_inverted"] = bool(inv)
            leads.append(robust_z(signal.resample_poly(y, 1, 4)))
            if q["ok"] and (best is None or q["tcorr"] > best[1]["tcorr"]):
                best = (pk, q, li)
        n125 = min(len(l) for l in leads)
        if best is not None:
            ihr = E.inst_hr(best[0], 500, n125, 125)
            out[f"{key}|rpeaks"] = best[0]
            row["ecg_lead_used"] = best[2] + 1
        else:
            ihr = np.full(n125, HR_MU)
        out[f"{key}|ecg"] = np.stack([leads[0][:n125], leads[1][:n125], (ihr - HR_MU) / HR_SD]).astype(np.float32)
        env, emg_f = SG.emg_envelopes(emg)
        out[f"{key}|emg"] = env
        row.update(SG.emg_quality(emg_f))
        wave, rate, rpk = SG.rsp_process(rsp)
        row.update(SG.rsp_quality(rate, rpk, dur))
        rate = np.where(np.isfinite(rate), rate, np.nanmedian(rate) if np.isfinite(rate).any() else BR_MU)
        out[f"{key}|rsp"] = np.stack([robust_z(wave), (rate - BR_MU) / BR_SD]).astype(np.float32)
        row.update(dur_s=dur, ecg_ok=best is not None, emg_ok=row["emg_flat"] < 0.5 and row["emg_extreme"] < 0.05)
        rows.append(row)
    np.savez_compressed(OUT / f"{sub}.npz", **out)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = sorted(p.stem for p in (DATA / "raw_segments").glob("*.npz"))
    if "--force" not in sys.argv and (DATA / "quality.csv").exists() and all((OUT / f"{s}.npz").exists() for s in subs):
        print("streams already pre-processed (use --force to redo)")
        print((DATA / "exclusion_reasons.yaml").read_text())
        return
    with Pool(10) as p:
        q = pd.DataFrame([r for rows in p.map(process, subs) for r in rows])
    q.to_csv(DATA / "quality.csv", index=False)
    seg = pd.read_csv(DATA / "segments.csv")
    excl, reasons = {"all": []}, {}
    for task, t in cfg["tasks"].items():
        need = [tuple(t["reference"])] + [tuple(s) for cls in t["segments"] for s in cls]
        bad = []
        for sub in subs:
            why = []
            for study, phase, level in need:
                r = q[(q.subject == sub) & (q.study == study) & (q.phase == phase) & (q.level == level)]
                exp = seg[(seg.study == study) & (seg.phase == phase) & (seg.level == level)].dur_s.median()
                name = f"{study} {phase} {level}"
                if r.empty or not r.present.iloc[0]:
                    why.append(f"{name}: missing")
                    continue
                r = r.iloc[0]
                if r.dur_s < 0.9 * exp:
                    why.append(f"{name}: {r.dur_s:.0f}s < 90% of {exp:.0f}s")
                if not r.ecg_ok:
                    why.append(f"{name}: no ECG lead passes SQI")
                if not r.rsp_ok:
                    why.append(f"{name}: respiration implausible ({r.rsp_rate:.1f}/min)")
                if not r.emg_ok:
                    why.append(f"{name}: EMG flat/saturated")
            if why:
                bad.append(sub)
                reasons.setdefault(task, {})[sub] = why
        excl[task] = bad
    (DATA / "exclusions.yaml").write_text(yaml.safe_dump(excl))
    (DATA / "exclusion_reasons.yaml").write_text(yaml.safe_dump(reasons, width=200))
    print("segments processed:", len(q), "| present:", int(q.present.sum()))
    print(q[["lead1_ok", "lead2_ok", "ecg_ok", "rsp_ok", "emg_ok"]].mean().round(3).to_string())
    for task, b in excl.items():
        if task != "all":
            print(f"{task}: {len(subs) - len(b)} subjects kept, excluded {b}")
    print(yaml.safe_dump(reasons, width=200))


if __name__ == "__main__":
    main()
