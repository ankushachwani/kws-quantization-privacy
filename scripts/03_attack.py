"""Run the attacks on the saved logits and bootstrap the results.

    python scripts/03_attack.py [--boot 1000]

Per model and bit-width: AUC, TPR@1% FPR, TPR@0.1% FPR for the loss / confidence / modified-entropy attacks, on a
balanced set (4,482 members vs 4,482 non-members) for each member draw (seed).

Bootstrap (B resamples): in resample b the non-member indices are drawn once and shared by all seeds and
bit-widths, each seed draws its own member indices (shared across bit-widths). The statistic is the mean over seeds.
Sharing indices across bit-widths makes the int8 vs w4a8 / w4a4 differences paired. The 95% CI is the 2.5 / 97.5
percentile of the B values. It covers sampling of clips, not the choice of seed set beyond the 10 seeds used.
"""
import argparse, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import *
from mia import *

ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--procs', type=int, default=5)
args = ap.parse_args()
draws = pd.read_csv(os.path.join(RAW, 'member_draws.csv'))
SEEDS = sorted(draws.seed.unique())[:args.seeds]


def load(key):
    out = {}
    uniq = None
    for var in VARIANTS:
        z = np.load(os.path.join(CACHE, f'logits_{key}_{var}.npz'))
        uniq = z['pool_idx']; ytest = z['test_label']; ymem = z['mem_label']
        keep = ytest != SILENCE
        out[var] = dict(test=z['test'][keep], mem=z['members'])
    # member index lists per seed, as positions in the union array
    pos = {s: np.searchsorted(uniq, draws[draws.seed == s].pool_idx.values) for s in SEEDS}
    return out, ytest[keep], ymem, pos


def run_model(key):
    L, ytest, ymem, pos = load(key)
    rng = np.random.RandomState(12345)
    # per variant, per attack: score vectors for all test (non-silence) and all union member clips
    S = {v: dict(test=scores(L[v]['test'], ytest), mem=scores(L[v]['mem'], ymem)) for v in VARIANTS}
    correct = {v: dict(test=(L[v]['test'].argmax(1) == ytest), mem=(L[v]['mem'].argmax(1) == ymem)) for v in VARIANTS}
    nn = len(ytest)

    def stat(v, a, ni, mi_by_seed):
        """mean over seeds of [auc, tpr1, tpr01] for resampled indices (ni test, mi per seed -> union positions)."""
        sn = S[v]['test'][a][ni]
        return np.mean([metrics(S[v]['mem'][a][mi], sn) for mi in mi_by_seed], 0)

    def gap(v, ni, mi_by_seed):
        return np.mean([correct[v]['mem'][mi].mean() for mi in mi_by_seed]) - correct[v]['test'][ni].mean()

    # point estimates per seed (no resampling)
    rows = []
    ident = np.arange(nn)
    for s in SEEDS:
        for v in VARIANTS:
            g = correct[v]['mem'][pos[s]].mean() - correct[v]['test'].mean()
            for a in ATTACKS:
                m = metrics(S[v]['mem'][a][pos[s]], S[v]['test'][a])
                rows.append(dict(model=key, variant=v, attack=a, seed=s, **dict(zip(METRIC_NAMES, m)),
                                 acc_member=correct[v]['mem'][pos[s]].mean(), acc_test=correct[v]['test'].mean(), acc_gap=g))
    pts = pd.DataFrame(rows)

    # temperature sensitivity of the loss attack (point estimates only)
    sens = []
    for T in (0.25, 1.0, 4.0):
        for v in VARIANTS:
            st = scores(L[v]['test'], ytest, T)['loss']; sm = scores(L[v]['mem'], ymem, T)['loss']
            for s in SEEDS: sens.append(dict(model=key, variant=v, T=T, seed=s, auc=auc(sm[pos[s]], st)))

    # bootstrap
    B = args.boot
    boot = {(v, a): np.zeros((B, 3)) for v in VARIANTS for a in ATTACKS}; bgap = {v: np.zeros(B) for v in VARIANTS}
    for b in range(B):
        ni = rng.randint(0, nn, nn)
        mis = [pos[s][rng.randint(0, len(pos[s]), len(pos[s]))] for s in SEEDS]
        for v in VARIANTS:
            bgap[v][b] = gap(v, ni, mis)
            for a in ATTACKS: boot[(v, a)][b] = stat(v, a, ni, mis)
    return key, pts, pd.DataFrame(sens), boot, bgap


if __name__ == '__main__':
    with Pool(min(args.procs, len(MODELS))) as p: res = p.map(run_model, MODELS)
    pts = pd.concat([r[1] for r in res]); pts.to_csv(os.path.join(RAW, 'per_seed_metrics.csv'), index=False)
    pd.concat([r[2] for r in res]).to_csv(os.path.join(RAW, 'temperature_sensitivity.csv'), index=False)
    lo, hi = 2.5, 97.5; main = []; diff = []; gaps = []
    for key, p_, _, boot, bgap in res:
        for v in VARIANTS:
            for a in ATTACKS:
                sub = p_[(p_.variant == v) & (p_.attack == a)]
                for j, mn in enumerate(METRIC_NAMES):
                    x = sub[mn].values; bb = boot[(v, a)][:, j]
                    main.append(dict(model=key, variant=v, attack=a, metric=mn, mean=x.mean(), sd_over_seeds=x.std(ddof=1), min_seed=x.min(), max_seed=x.max(),
                                     ci_lo=np.percentile(bb, lo), ci_hi=np.percentile(bb, hi), n_boot=len(bb), n_seeds=len(x)))
                    if v != 'int8':
                        d = bb - boot[('int8', a)][:, j]; d0 = x.mean() - p_[(p_.variant == 'int8') & (p_.attack == a)][mn].values.mean()
                        diff.append(dict(model=key, attack=a, metric=mn, contrast=f'{v} - int8', diff=d0, ci_lo=np.percentile(d, lo), ci_hi=np.percentile(d, hi),
                                         ci_excludes_0=bool(np.percentile(d, lo) > 0 or np.percentile(d, hi) < 0)))
            sub = p_[(p_.variant == v) & (p_.attack == 'loss')]
            gaps.append(dict(model=key, variant=v, acc_member=sub.acc_member.mean(), acc_test=sub.acc_test.iloc[0], acc_gap=sub.acc_gap.mean(),
                             gap_sd_over_seeds=sub.acc_gap.std(ddof=1), gap_ci_lo=np.percentile(bgap[v], lo), gap_ci_hi=np.percentile(bgap[v], hi)))
        # raw bootstrap draws, one file per model (long format)
        out = []
        for v in VARIANTS:
            for a in ATTACKS:
                d = pd.DataFrame(boot[(v, a)], columns=METRIC_NAMES); d['variant'] = v; d['attack'] = a; d['gap'] = bgap[v]; d['b'] = np.arange(len(d)); out.append(d)
        pd.concat(out).to_csv(os.path.join(RAW, f'bootstrap_{key}.csv.gz'), index=False)
    pd.DataFrame(main).to_csv(os.path.join(RAW, 'summary_long.csv'), index=False)
    pd.DataFrame(diff).to_csv(os.path.join(RAW, 'paired_differences.csv'), index=False)
    pd.DataFrame(gaps).to_csv(os.path.join(RAW, 'accuracy_gap.csv'), index=False)
    print('done')
