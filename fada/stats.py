import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def friedman(table):
    """table: (n_blocks, k_conditions). Returns chi2, p, Kendall's W."""
    table = np.asarray(table, float)
    n, k = table.shape
    chi2, p = stats.friedmanchisquare(*table.T)
    return dict(chi2=chi2, p=p, kendall_w=chi2 / (n * (k - 1)), n=n, k=k)


def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def pairwise_wilcoxon(df, pairs=None, method="holm"):
    """df: blocks x conditions. Paired two-sided Wilcoxon with Holm correction."""
    cols = list(df.columns)
    pairs = pairs or list(itertools.combinations(cols, 2))
    rows = []
    for a, b in pairs:
        x, y = df[a].values, df[b].values
        try:
            p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue if np.any(x != y) else 1.0
        except ValueError:
            p = 1.0
        rows.append(dict(a=a, b=b, mean_a=x.mean(), mean_b=y.mean(), diff=np.mean(x - y), p=p, r_rb=rank_biserial(x, y)))
    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out.p, method=method)[1]
    return out


def nemenyi_cd(k, n, alpha=0.05):
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def mean_ranks(table, higher_better=True):
    t = -np.asarray(table) if higher_better else np.asarray(table)
    return np.mean([stats.rankdata(r) for r in t], 0)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    rng = np.random.RandomState(seed)
    bs = v[rng.randint(0, len(v), (n_boot, len(v)))].mean(1)
    return v.mean(), np.percentile(bs, 100 * alpha / 2), np.percentile(bs, 100 * (1 - alpha / 2))
