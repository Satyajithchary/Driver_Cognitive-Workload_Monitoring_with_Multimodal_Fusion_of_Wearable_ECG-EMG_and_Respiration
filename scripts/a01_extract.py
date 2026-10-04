"""Extracts, for every subject, every annotated segment (STUDY, PHASE, LEVEL) of
the n-back and k-drive studies with the raw ECG I/II, RSP and EMG at their native
bandwidth (ECG 500 Hz, RSP 250 Hz, EMG 1000 Hz), plus META / PERFORMANCE /
SUBJECTIVE tables."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fada.common import load_cfg

cfg = load_cfg()
RAW = Path(cfg["paths"]["raw"])
OUT = Path(cfg["paths"]["data"]) / "raw_segments"
COLS = ["RAW_ECG_I", "RAW_ECG_II", "RAW_RSP", "RAW_EMG", "STUDY", "PHASE", "LEVEL"]


def run(path):
    sub = path.stem
    out = OUT / f"{sub}.npz"
    if out.exists():
        return sub, None
    # chunked read: PyTables loads whole rows (149 columns), so select per chunk
    with pd.HDFStore(path, "r") as st:
        d = pd.concat([c[COLS] for c in st.select("SIGNALS", chunksize=1_000_000)])
    rows, arrays = [], {}
    for (study, phase, level), g in d.groupby(["STUDY", "PHASE", "LEVEL"], sort=False):
        if study not in ("n-back", "k-drive") or phase not in ("baseline", "test", "train"):
            continue
        idx = g.index.values
        # a segment must be one contiguous block
        breaks = np.where(np.diff(idx) != 1)[0]
        blocks = np.split(np.arange(len(idx)), breaks + 1)
        blk = max(blocks, key=len)
        g = g.iloc[blk]
        key = f"{study}|{phase}|{int(level)}"
        nan = g[["RAW_ECG_I", "RAW_ECG_II", "RAW_RSP", "RAW_EMG"]].isna().mean()
        x = g[["RAW_ECG_I", "RAW_ECG_II", "RAW_RSP", "RAW_EMG"]].interpolate(limit_direction="both").values
        arrays[key + "|ecg"] = signal.resample_poly(x[:, :2], 1, 2, axis=0).astype(np.float32)  # 500 Hz
        arrays[key + "|rsp"] = signal.resample_poly(x[:, 2], 1, 4).astype(np.float32)           # 250 Hz
        arrays[key + "|emg"] = x[:, 3].astype(np.float32)                                        # 1000 Hz
        rows.append(dict(subject=sub, study=study, phase=phase, level=int(level), start_ms=int(idx[blk[0]]),
                         dur_s=len(g) / 1000, n_blocks=len(blocks), nan_ecg=nan.iloc[0], nan_rsp=nan.iloc[2],
                         nan_emg=nan.iloc[3]))
    np.savez_compressed(out, **arrays)
    st = pd.HDFStore(path, "r")
    tabs = {k: st[k].assign(subject=sub) for k in ("META", "PERFORMANCE", "SUBJECTIVE")}
    st.close()
    return sub, (pd.DataFrame(rows), tabs)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.hdf5"))
    print(len(files), "subject files", flush=True)
    seg, tabs = [], {"META": [], "PERFORMANCE": [], "SUBJECTIVE": []}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        for i, (sub, res) in enumerate(p.imap_unordered(run, files)):
            print(f"[{i + 1}/{len(files)}] {sub} {'cached' if res is None else len(res[0])} segments", flush=True)
            if res is not None:
                seg.append(res[0])
                for k, v in res[1].items():
                    tabs[k].append(v)
    d = Path(cfg["paths"]["data"])
    if seg:
        old = pd.read_csv(d / "segments.csv") if (d / "segments.csv").exists() else pd.DataFrame()
        pd.concat([old] + seg).drop_duplicates(["subject", "study", "phase", "level"], keep="last").to_csv(d / "segments.csv", index=False)
        for k, v in tabs.items():
            if v:
                frame = pd.concat(v)
                frame.columns = [" ".join(c) if isinstance(c, tuple) else c for c in frame.columns]
                f = d / f"{k.lower()}.csv"
                frame.to_csv(f, mode="a" if f.exists() else "w", header=not f.exists(), index=False)
