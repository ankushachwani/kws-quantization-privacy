"""Tables (csv + md) and figures from results/raw and the cached logits."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from common import *
from mia import scores, ATTACKS

LABEL = {'s': 'DS-CNN-S', 'm': 'DS-CNN-M', 'l': 'DS-CNN-L', 'tc8': 'TC-ResNet8', 'tc14_15': 'TC-ResNet14-1.5'}
summ = pd.read_csv(os.path.join(RAW, 'summary_long.csv')); gaps = pd.read_csv(os.path.join(RAW, 'accuracy_gap.csv'))
diff = pd.read_csv(os.path.join(RAW, 'paired_differences.csv')); pts = pd.read_csv(os.path.join(RAW, 'per_seed_metrics.csv'))


def save(df, name, md_floatfmt=None):
    df.to_csv(os.path.join(TABLES, name + '.csv'), index=False)
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for r in df.itertuples(index=False): lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    open(os.path.join(TABLES, name + '.md'), 'w').write('\n'.join(lines) + '\n')


def cell(r, pct=False, d=3):
    f = (lambda x: f'{100 * x:.{d - 1}f}') if pct else (lambda x: f'{x:.{d}f}')
    return f'{f(r["mean"])} [{f(r.ci_lo)}, {f(r.ci_hi)}]'


def get(model, var, attack, metric):
    return summ[(summ.model == model) & (summ.variant == var) & (summ.attack == attack) & (summ.metric == metric)].iloc[0]


# ---- main table: loss attack -------------------------------------------------------------------
rows = []
for m in MODELS:
    for v in VARIANTS:
        g = gaps[(gaps.model == m) & (gaps.variant == v)].iloc[0]
        a = get(m, v, 'loss', 'auc'); t1 = get(m, v, 'loss', 'tpr_1pct'); t01 = get(m, v, 'loss', 'tpr_0.1pct')
        rows.append({'model': LABEL[m], 'bits': v, 'test acc %': f'{100 * g.acc_test:.2f}', 'member acc %': f'{100 * g.acc_member:.2f}',
                     'gap (pp) [95% CI]': f'{100 * g.acc_gap:.2f} [{100 * g.gap_ci_lo:.2f}, {100 * g.gap_ci_hi:.2f}]',
                     'AUC [95% CI]': cell(a), 'AUC sd over seeds': f'{a.sd_over_seeds:.4f}',
                     'TPR@1% FPR % [95% CI]': cell(t1, True, 2), 'TPR@0.1% FPR % [95% CI]': cell(t01, True, 2)})
save(pd.DataFrame(rows), 'main_loss_attack')

# ---- other attacks, AUC ------------------------------------------------------------------------
rows = []
for m in MODELS:
    for v in VARIANTS:
        r = {'model': LABEL[m], 'bits': v}
        for a in ATTACKS:
            for mn, nm in (('auc', 'AUC'), ('tpr_1pct', 'TPR@1%'), ('tpr_0.1pct', 'TPR@0.1%')):
                x = get(m, v, a, mn); r[f'{a} {nm}'] = f'{x["mean"]:.3f} [{x.ci_lo:.3f}, {x.ci_hi:.3f}]' if mn == 'auc' else f'{100 * x["mean"]:.2f} [{100 * x.ci_lo:.2f}, {100 * x.ci_hi:.2f}]'
        rows.append(r)
save(pd.DataFrame(rows), 'all_attacks')

# ---- paired differences ------------------------------------------------------------------------
d = diff.copy()
for c in ('diff', 'ci_lo', 'ci_hi'): d[c] = d[c].map(lambda x: round(x, 4))
d['model'] = d.model.map(LABEL)
save(d.rename(columns={'diff': 'difference (seed mean)'}), 'paired_differences')

# ---- gap vs leakage ----------------------------------------------------------------------------
rows = []
x = gaps.set_index(['model', 'variant'])
auc_l = summ[(summ.attack == 'loss') & (summ.metric == 'auc')].set_index(['model', 'variant'])['mean']
g = gaps.assign(auc=[auc_l[(r.model, r.variant)] for r in gaps.itertuples()])
cor = []
for name, sub in [('all 15 (model, bits) points', g)] + [(f'{v} only (5 models)', g[g.variant == v]) for v in VARIANTS]:
    rs, ps = spearmanr(sub.acc_gap, sub.auc); rp, pp = pearsonr(sub.acc_gap, sub.auc)
    cor.append({'subset': name, 'n': len(sub), 'spearman': round(rs, 3), 'spearman p': round(ps, 4), 'pearson': round(rp, 3), 'pearson p': round(pp, 4)})
save(pd.DataFrame(cor), 'gap_vs_auc_correlation')

# ---- figures -----------------------------------------------------------------------------------
COL = {'int8': '#1b6ca8', 'w4a8': '#e08a00', 'w4a4': '#b3261e'}
fpr_grid = np.logspace(np.log10(1 / 4482), 0, 200)


def roc_curve(sm, sn):
    """TPR at each FPR on the grid with the same strict-threshold rule as mia.tpr_at_fpr."""
    n = len(sn); srt = np.sort(sn)
    k = np.floor(fpr_grid * n).astype(int).clip(0, n - 1)
    thr = srt[n - k - 1]
    return np.array([(sm > t).mean() for t in thr])


draws = pd.read_csv(os.path.join(RAW, 'member_draws.csv')); seeds = sorted(draws.seed.unique())
fig, axs = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
for ax, m in zip(axs, MODELS):
    for v in VARIANTS:
        z = np.load(os.path.join(CACHE, f'logits_{m}_{v}.npz')); keep = z['test_label'] != SILENCE
        sn = scores(z['test'][keep], z['test_label'][keep])['loss']; sm_all = scores(z['members'], z['mem_label'])['loss']
        curves = []
        for s in seeds:
            pos = np.searchsorted(z['pool_idx'], draws[draws.seed == s].pool_idx.values); curves.append(roc_curve(sm_all[pos], sn))
        curves = np.array(curves)
        ax.fill_between(fpr_grid, curves.min(0), curves.max(0), color=COL[v], alpha=0.18, lw=0)
        a = get(m, v, 'loss', 'auc')
        ax.plot(fpr_grid, curves.mean(0), color=COL[v], lw=1.6, label=f'{v}  AUC {a["mean"]:.3f}')
    ax.plot([1e-4, 1], [1e-4, 1], 'k:', lw=1)
    for f in (0.001, 0.01): ax.axvline(f, color='gray', lw=0.6, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1 / 4482 * 0.9, 1); ax.set_ylim(1e-4, 1)
    ax.set_title(LABEL[m]); ax.set_xlabel('FPR'); ax.legend(loc='lower right', fontsize=8)
axs[0].set_ylabel('TPR')
fig.suptitle('Loss-threshold attack, members = assumed training clips vs official test clips. Line = mean over 10 member draws, band = min..max. Dashed verticals: 0.1% and 1% FPR', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'roc_loglog.png'), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.4))
mk = dict(zip(MODELS, 'osD^v'))
for r in g.itertuples():
    ax.errorbar(100 * r.acc_gap, r.auc, xerr=[[100 * (r.acc_gap - r.gap_ci_lo)], [100 * (r.gap_ci_hi - r.acc_gap)]], fmt=mk[r.model], color=COL[r.variant], ms=6, lw=0.8)
ax.axhline(0.5, color='k', ls=':', lw=1); ax.axvline(0, color='k', ls=':', lw=1)
ax.set_xlabel('member acc - test acc (percentage points)'); ax.set_ylabel('loss-attack AUC')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk[m], color='gray', ls='', label=LABEL[m]) for m in MODELS] + [Line2D([], [], marker='o', color=COL[v], ls='', label=v) for v in VARIANTS]
ax.legend(handles=h, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'gap_vs_auc.png'), dpi=160)
print('tables and figures written')
