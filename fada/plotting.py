from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = {"Normal": SLOTS[0], "Cognitive": SLOTS[1], "Emotional": SLOTS[2]}
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"

SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb"] + SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b22d2c"])

FAMILY = {
    "ecg_only": "Unimodal", "video_only": "Unimodal", "rppg_only": "Unimodal", "emg_only": "Unimodal", "rsp_only": "Unimodal",
    "early": "Early",
    "inter_concat": "Intermediate", "inter_attn": "Intermediate",
    "late": "Late", "late_xattn": "Late",
    "tfn": "Tensor", "lmf": "Tensor", "hot": "Tensor",
    "hierarchical": "Hierarchical",
}
FAMILY_COLORS = {"Unimodal": GRAY, "Early": SLOTS[0], "Intermediate": SLOTS[1], "Late": SLOTS[2],
                 "Tensor": SLOTS[6], "Hierarchical": SLOTS[4]}
LABELS = {
    "ecg_only": "ECG only", "video_only": "Video only", "rppg_only": "rPPG only", "emg_only": "EMG only", "rsp_only": "RSP only",
    "early": "Early (joint tokens)",
    "inter_concat": "Intermediate-Concat", "inter_attn": "Intermediate-Attention",
    "late": "Late (decision)", "late_xattn": "Late cross-attention",
    "tfn": "Bilinear (TFN)", "lmf": "Low-rank bilinear (LMF)", "hot": "Higher-order (HOT-3)",
    "hierarchical": "Hierarchical",
}
SHORT = {
    "ecg_only": "ECG", "video_only": "Video", "rppg_only": "rPPG", "emg_only": "EMG", "rsp_only": "RSP",
    "early": "Early", "inter_concat": "Int-Cat",
    "inter_attn": "Int-Att", "late": "Late", "late_xattn": "Late-XA", "tfn": "TFN",
    "lmf": "LMF", "hot": "HOT", "hierarchical": "Hier",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "font.family": "DejaVu Sans",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "lines.linewidth": 2, "axes.prop_cycle": plt.cycler(color=SLOTS),
        "savefig.bbox": "tight", "figure.facecolor": "white",
    })


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def family(name):
    return FAMILY.get(name.split("@")[0], "Unimodal")


def short(name):
    if "@" in name:
        f, m = name.split("@")
        return SHORT[f] + "(" + "+".join(x[0].upper() for x in m.split("+")) + ")"
    return SHORT.get(name, name)


def label(name):
    if "@" in name:
        f, m = name.split("@")
        return f"{LABELS[f]} [{' + '.join(x.upper() for x in m.split('+'))}]"
    return LABELS.get(name, name)
