"""Training-identical MFCC front end (torchaudio), applied clip by clip so a clip's features never depend on
which other clips share its batch (torchaudio's top_db floor is otherwise taken over the whole batch)."""
import numpy as np, torch, torchaudio

SR = 16000
_MT = {}

def mfcc_transform(kind):
    if kind not in _MT:
        if kind == 'tang':
            _MT[kind] = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=40, melkwargs=dict(
                n_fft=480, win_length=480, hop_length=160, f_min=20.0, f_max=4000.0, n_mels=40, center=True))
        else:
            _MT[kind] = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=10, melkwargs=dict(
                n_fft=640, win_length=640, hop_length=320, f_min=20.0, f_max=4000.0, n_mels=40, center=False))
    return _MT[kind]

def features(waves_i16, kind, mu, sd):
    """waves_i16: N x 16000 int16 -> N x T x n_mfcc float32, normalised with the training mean/std."""
    mt = mfcc_transform(kind); out = []
    with torch.no_grad():
        for w in waves_i16:
            f = mt(torch.from_numpy(w.astype(np.float32) / 32768.0).unsqueeze(0))     # 1 x n_mfcc x T
            out.append(((f.transpose(1, 2) - torch.from_numpy(mu).view(1, 1, -1)) / torch.from_numpy(sd).view(1, 1, -1))[0].numpy())
    return np.stack(out).astype(np.float32)
