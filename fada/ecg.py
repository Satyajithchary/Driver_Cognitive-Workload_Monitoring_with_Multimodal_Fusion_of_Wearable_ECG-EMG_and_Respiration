import numpy as np
import neurokit2 as nk
from scipy import signal, stats

SAT_LO, SAT_HI = 50, 65480  # 16-bit ADC rails


def fix_saturation(x):
    bad = (x <= SAT_LO) | (x >= SAT_HI)
    if bad.any() and (~bad).sum() > 10:
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x, float(bad.mean())


def bandpass(x, fs, band, order=4):
    sos = signal.butter(order, band, btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def r_peaks(x, fs):
    try:
        _, info = nk.ecg_peaks(x, sampling_rate=fs, method="neurokit", correct_artifacts=True)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def sqi(x, fs, peaks=None):
    """Signal quality: kurtosis SQI, template correlation, HR plausibility."""
    if peaks is None:
        peaks = r_peaks(x, fs)
    out = {"ksqi": float(stats.kurtosis(x)), "n_peaks": int(len(peaks))}
    if len(peaks) < 10:
        out.update(hr=np.nan, rr_cv=np.nan, tcorr=0.0, ok=False)
        return out
    rr = np.diff(peaks) / fs
    w = int(0.25 * fs)
    beats = np.stack([x[p - w:p + w] for p in peaks if p - w >= 0 and p + w < len(x)])
    tmpl = np.median(beats, 0)
    tc = np.median([np.corrcoef(b, tmpl)[0, 1] for b in beats])
    hr = 60 / np.median(rr)
    # fraction of the record explained by detected beats; periodic noise gives
    # a plausible median RR but far too few beats
    cov = len(peaks) * np.median(rr) / (len(x) / fs)
    out.update(hr=float(hr), rr_cv=float(np.std(rr) / np.mean(rr)), tcorr=float(tc), coverage=float(cov))
    out["ok"] = bool(40 <= hr <= 160 and tc > 0.8 and out["rr_cv"] < 0.3 and 0.85 < cov < 1.15)
    return out


def hrv_features(peaks, fs, t0=None, t1=None):
    """Time-domain HRV in a window (seconds); peaks in samples."""
    p = peaks / fs
    if t0 is not None:
        p = p[(p >= t0) & (p < t1)]
    if len(p) < 4:
        return dict(hr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
    rr = np.diff(p) * 1000
    rr = rr[(rr > 300) & (rr < 1600)]
    if len(rr) < 3:
        return dict(hr=np.nan, sdnn=np.nan, rmssd=np.nan, pnn50=np.nan)
    d = np.diff(rr)
    return dict(hr=60000 / rr.mean(), sdnn=rr.std(ddof=1), rmssd=np.sqrt(np.mean(d ** 2)),
                pnn50=100 * np.mean(np.abs(d) > 50))


def inst_hr(peaks, fs_peaks, n_out, fs_out):
    """Instantaneous HR (bpm) on the output grid: RR at each beat, implausible
    intervals dropped, linear interpolation, edge hold."""
    t = peaks[1:] / fs_peaks
    rr = np.diff(peaks) / fs_peaks
    ok = (rr > 0.3) & (rr < 1.6)
    # drop beats that jump >30 % from the local median (missed / extra beats)
    med = signal.medfilt(rr, 7)
    ok &= np.abs(rr - med) < 0.3 * med
    grid = np.arange(n_out) / fs_out
    return np.interp(grid, t[ok], 60 / rr[ok]).astype(np.float32)


def preprocess(x, fs_raw, fs_out, band, clip_z):
    x, sat = fix_saturation(x)
    y = bandpass(x, fs_raw, band)
    y, inverted = nk.ecg_invert(y, sampling_rate=fs_raw)
    peaks = r_peaks(y, fs_raw)
    q = sqi(y, fs_raw, peaks)
    z = signal.resample_poly(y, fs_out, fs_raw)
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med)) + 1e-8
    z = np.clip((z - med) / mad, -clip_z, clip_z).astype(np.float32)
    q.update(sat_frac=sat, inverted=bool(inverted))
    return z, inst_hr(peaks, fs_raw, len(z), fs_out), peaks, q
