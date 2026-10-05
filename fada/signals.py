"""EMG and respiration processing plus simple quality indices."""
import numpy as np
import neurokit2 as nk
from scipy import signal

EMG_BANDS = [(20, 60), (60, 150), (150, 450)]


def emg_envelopes(x, fs=1000, fs_out=50, win_s=0.05):
    """Band-pass 20-450 Hz, notch 50 Hz + harmonics, then log-RMS envelope of three
    sub-bands (low / mid / high frequency content) on a fs_out grid."""
    x = signal.sosfiltfilt(signal.butter(4, [20, 450], "band", fs=fs, output="sos"), x)
    for f0 in (50, 100, 150, 200, 250, 300, 350, 400):
        b, a = signal.iirnotch(f0, 30, fs)
        x = signal.filtfilt(b, a, x)
    step, L = fs // fs_out, int(win_s * fs)
    out = []
    for lo, hi in EMG_BANDS:
        y = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=fs, output="sos"), x)
        p = np.convolve(y ** 2, np.ones(L) / L, mode="same")[::step]
        out.append(0.5 * np.log(p + 1e-12))
    return np.stack(out).astype(np.float32), x


def emg_quality(x_filt, fs=1000):
    """Fraction of flat (dead) samples and of saturated / extreme samples."""
    d = np.abs(np.diff(x_filt))
    flat = np.mean(d < 1e-7)
    z = np.abs((x_filt - np.median(x_filt)) / (1.4826 * np.median(np.abs(x_filt - np.median(x_filt))) + 1e-12))
    return dict(emg_flat=float(flat), emg_extreme=float(np.mean(z > 20)), emg_rms=float(np.sqrt(np.mean(x_filt ** 2))))


def rsp_process(x, fs=250, fs_out=25):
    """Band-pass 0.05-1 Hz, peaks (breaths), instantaneous breathing rate (bpm)."""
    y = signal.sosfiltfilt(signal.butter(2, [0.05, 1.0], "band", fs=fs, output="sos"), x)
    try:
        _, info = nk.rsp_peaks(y, sampling_rate=fs, method="khodadad2018")
        pk = np.asarray(info["RSP_Peaks"], int)
    except Exception:
        pk = np.array([], int)
    z = signal.resample_poly(y, fs_out, fs)
    t = np.arange(len(z)) / fs_out
    if len(pk) >= 3:
        tp = pk[1:] / fs
        br = 60 / (np.diff(pk) / fs)
        ok = (br > 4) & (br < 40)
        rate = np.interp(t, tp[ok], br[ok]) if ok.sum() >= 2 else np.full(len(t), np.nan)
    else:
        rate = np.full(len(t), np.nan)
    return z.astype(np.float32), rate.astype(np.float32), pk


def rsp_quality(rate, pk, dur_s):
    br = 60 * len(pk) / dur_s if dur_s > 0 else np.nan
    return dict(rsp_rate=float(br), rsp_ok=bool(4 <= br <= 40 and np.isfinite(rate).mean() > 0.9))
