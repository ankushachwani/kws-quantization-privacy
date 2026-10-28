"""Scores and metrics for the membership attacks. Higher score = predicted member."""
import numpy as np
from scipy.stats import rankdata

T_MAIN = 1.0       # softmax temperature applied to the float logits that run_int returns (int accumulator * fscale + fbias)
ATTACKS = ['loss', 'conf', 'ment']
ALPHAS = (0.01, 0.001)


def softmax(logits, T=T_MAIN):
    z = logits.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def scores(logits, y, T=T_MAIN):
    """Returns dict attack -> score vector.
    loss : -cross-entropy of the true label (Yeom et al. loss threshold)
    conf : max softmax probability (does not use the label)
    ment : minus the modified entropy of Song & Mittal 2021, which uses the true label y
    """
    z = logits.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(logp)
    n = np.arange(len(y))
    py = p[n, y]
    eps = 1e-30
    # -(1-p_y) log p_y - sum_{i != y} p_i log(1 - p_i)
    other = -(p * np.log(np.clip(1 - p, eps, 1))).sum(1) + py * np.log(np.clip(1 - py, eps, 1))
    ment = -(1 - py) * logp[n, y] + other
    return dict(loss=logp[n, y], conf=p.max(1), ment=-ment)


def auc(s_mem, s_non):
    r = rankdata(np.concatenate([s_mem, s_non]))
    nm, nn = len(s_mem), len(s_non)
    return (r[:nm].sum() - nm * (nm + 1) / 2) / (nm * nn)


def tpr_at_fpr(s_mem, s_non, alpha):
    """TPR when the threshold is the (k+1)-th largest non-member score, k = floor(alpha * n_non); members must
    score strictly above it. Ties therefore count against the attacker, FPR is <= alpha."""
    n = len(s_non); k = int(np.floor(alpha * n))
    thr = np.partition(s_non, n - k - 1)[n - k - 1]
    return float((s_mem > thr).mean())


def metrics(s_mem, s_non):
    return [auc(s_mem, s_non)] + [tpr_at_fpr(s_mem, s_non, a) for a in ALPHAS]
METRIC_NAMES = ['auc', 'tpr_1pct', 'tpr_0.1pct']
