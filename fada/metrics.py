import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


def ece(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return out


def classification(prob, y, n_cls=3):
    pred = prob.argmax(1)
    out = dict(
        acc=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        f1_macro=f1_score(y, pred, average="macro", labels=range(n_cls), zero_division=0),
        kappa=cohen_kappa_score(y, pred),
        mcc=matthews_corrcoef(y, pred),
        ece=ece(prob, y),
        nll=float(-np.log(prob[np.arange(len(y)), y] + 1e-12).mean()),
    )
    try:
        out["auroc"] = roc_auc_score(y, prob[:, 1]) if n_cls == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auroc"] = np.nan
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(n_cls), zero_division=0)
    for k in range(n_cls):
        out[f"prec_{k}"], out[f"rec_{k}"], out[f"f1_{k}"] = p[k], r[k], f[k]
    return out


def recording_level(prob, y, rec):
    """Average window probabilities within each 10-min recording."""
    recs = np.unique(rec)
    P = np.stack([prob[rec == r].mean(0) for r in recs])
    Y = np.array([y[rec == r][0] for r in recs])
    return P, Y, recs


def confusion(prob, y, n_cls=3):
    return confusion_matrix(y, prob.argmax(1), labels=range(n_cls))
