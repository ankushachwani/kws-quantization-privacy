"""Draw the member sets, run every (model, variant) with the bit-exact integer simulator, save logits.

    python scripts/02_logits.py [--seeds 10]

Member draw for seed k: for every non-silence class c, sample n_c clips without replacement from the training-split
pool of class c, where n_c = number of non-silence test clips of class c. Class 'unknown' is sampled uniformly from
the 25 non-keyword words. Test clips are the non-members.

Logits are the float32 head output of qmodel.run_int (int32 accumulator * fscale + fbias); no softmax here.
Check on the official test set: predictions are compared with the stored cryptonite simulation (*_sim.npz) and eval
accuracy with the model json. Features use recovered mu/sd (00_recover_norm.py), so a clip or two may flip.
"""
import argparse, os, time, json
import numpy as np, pandas as pd
from common import *
import qmodel as Q
from blob import load_blob, quant_input
from frontend import features

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--models', default=','.join(MODELS))
args = ap.parse_args()
SC = os.path.join(DATA, 'sc_v2'); TEST = os.path.join(DATA, 'test_set')
pool = pd.read_csv(os.path.join(CACHE, 'train_pool.csv')); te = pd.read_csv(os.path.join(CACHE, 'test_index.csv'))
man = pd.read_csv(os.path.join(ASSETS, 'manifest.csv'))
assert (man['class'].values == np.array(CLASSES)[te.label.values]).all(), 'manifest order differs from test index'
ev = (man.role == 'eval').values

# ---- member draws -------------------------------------------------------------------------------
need = te[te.label != SILENCE].label.value_counts().sort_index()
draws = []
for s in range(args.seeds):
    rng = np.random.RandomState(1000 + s)
    for c, n in need.items():
        idx = np.where(pool.label.values == c)[0]
        draws += [(s, int(i)) for i in rng.choice(idx, n, replace=False)]
draws = pd.DataFrame(draws, columns=['seed', 'pool_idx'])
draws.to_csv(os.path.join(RAW, 'member_draws.csv'), index=False)
uniq = np.sort(draws.pool_idx.unique()); print('member draws: %d seeds, %d unique clips' % (args.seeds, len(uniq)))

# ---- waves --------------------------------------------------------------------------------------
wt_path = os.path.join(CACHE, 'waves_test.npy')
if not os.path.exists(wt_path):
    np.save(wt_path, np.stack([read_wav16k(os.path.join(TEST, r.folder, r.file)) for r in te.itertuples()]))
W_test = np.load(wt_path)
wm_path = os.path.join(CACHE, 'waves_pool_%d.npy' % args.seeds)
if not os.path.exists(wm_path):
    np.save(wm_path, np.stack([read_wav16k(os.path.join(SC, pool.rel[i])) for i in uniq]))
W_mem = np.load(wm_path)

def run(net, qops, xq, bs=128):
    return np.concatenate([Q.run_int(net, qops, xq[i:i + bs])[1] for i in range(0, len(xq), bs)])

for key in args.models.split(','):
    kind = json.load(open(os.path.join(ASSETS, 'models', 'int8', key + '.json')))['frontend']
    mu, sd = load_norm(kind)
    Xt, Xm = features(W_test, kind, mu, sd), features(W_mem, kind, mu, sd)
    for var in VARIANTS:
        t0 = time.time(); base = os.path.join(ASSETS, 'models', var, key)
        js = json.load(open(base + '.json')); net, qops, _, _ = load_blob(base + '.bin')
        hi0 = 7 if var == 'w4a4' else 127          # same input clip as lowbit.py: w4a4 inputs are 4-bit
        lt = run(net, qops, quant_input(Xt, js['s_in'], net, hi0))
        sim = np.load(base + '_sim.npz'); pred = lt.argmax(1)
        agree = (pred == sim['pred']).mean(); acc = (pred[ev] == te.label.values[ev]).mean()
        ref = js['acc'] if var != 'int8' else js['int8_acc']
        print('%-8s %-5s test-set check: pred==stored sim %.4f (%d clips differ), eval acc %.4f (json %.4f)'
              % (key, var, agree, (pred != sim['pred']).sum(), acc, ref))
        lm = run(net, qops, quant_input(Xm, js['s_in'], net, hi0))
        np.savez_compressed(os.path.join(CACHE, f'logits_{key}_{var}.npz'), test=lt, members=lm, pool_idx=uniq,
                            test_label=te.label.values, mem_label=pool.label.values[uniq], test_pred_sim=sim['pred'])
        print('   %.0fs' % (time.time() - t0), flush=True)
