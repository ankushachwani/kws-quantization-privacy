"""Recover the MFCC mean/std the models were trained with.

The cryptonite repo built its feature cache from cryptonite_kws/weights/{s,tc8}.npz (keys mu, sd), but that folder
is not in the tree I was given and the model json/blobs do not store mu/sd. The cached features are an affine
function of the raw MFCC, feats = (raw - mu) / sd per coefficient, so I fit that line per coefficient on the
4,890 test clips and invert it. The residual is printed; it should be at float32 rounding level.

    python scripts/00_recover_norm.py /path/to/cryptonite/.cache
Writes assets/norm_{hello,tang}.json. Only needed once; the output is committed.
"""
import sys, os, json
import numpy as np
from common import *
from frontend import mfcc_transform

cache = sys.argv[1]
waves = np.load(os.path.join(cache, 'waves.npy'))
ncoef = {'hello': 10, 'tang': 40}
for kind in ('hello', 'tang'):
    F = np.load(os.path.join(cache, f'feats_{kind}.npy'))                       # N x T x C
    mt = mfcc_transform(kind); raw = []
    with torch.no_grad():
        for w in waves:
            raw.append(mt(torch.from_numpy(w.astype(np.float32) / 32768.0).unsqueeze(0))[0].T.numpy())
    R = np.stack(raw).astype(np.float64); F = F.astype(np.float64)
    mu = np.zeros(ncoef[kind]); sd = np.zeros(ncoef[kind]); worst = 0
    for k in range(ncoef[kind]):
        a, b = np.polyfit(R[:, :, k].ravel(), F[:, :, k].ravel(), 1)
        sd[k], mu[k] = 1 / a, -b / a
        worst = max(worst, np.abs(a * R[:, :, k] + b - F[:, :, k]).max())
    print(kind, 'max |fit - cached feature| =', worst)
    json.dump(dict(kind=kind, mu=mu.tolist(), sd=sd.tolist(), max_fit_residual=worst,
                   note='recovered by regression from cryptonite .cache/feats_%s.npy, not read from the training weights' % kind),
              open(os.path.join(ASSETS, f'norm_{kind}.json'), 'w'), indent=1)
