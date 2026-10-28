"""int8 post-training quantiser + bit-exact numpy reference + on-device blob writer for the KWS model zoo.

Pipeline for one model key:
    float IR (BN folded, ops in the order the device runs them)        build_float_ir()
 -> calibration on the 'calib' clips (activation ranges)                 calibrate()
 -> integer IR (int8 weights, int32 bias, fixed-point requant)           quantize()
 -> numpy integer executor (exactly the arithmetic the ESP32 runs)       run_int()
 -> binary blob parsed by firmware/src/kws_engine.cpp                    write_blob()

Quantisation scheme (TFLite-like, symmetric, zero-point 0):
  * weights  int8 per output channel;  activations int8 (signed tensors) or uint8 (post-ReLU tensors), one scale per tensor
  * conv/fc epilogue: out = clip(((relu?max(acc,0):acc) * mult[c] + 2^(sh-1)) >> sh[c]  +  boff[c], lo, hi)
    (boff carries the BatchNorm shift of the res-family, where BN sits after ReLU and cannot be folded into the conv)
"""
import math, struct, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------
# op / tensor codes shared with firmware/src/kws_engine.h
OP_CONV, OP_FC, OP_ADD, OP_AFFINE, OP_GAPFC = 0, 1, 2, 3, 4
OPNAME = {OP_CONV: 'conv', OP_FC: 'fc', OP_ADD: 'add', OP_AFFINE: 'affine', OP_GAPFC: 'gapfc'}


class Tn:                      # tensor descriptor (NHWC, batch 1 on device)
    def __init__(s, h, w, c, signed): s.h, s.w, s.c, s.signed = h, w, c, signed; s.scale = None
    @property
    def n(s): return s.h * s.w * s.c


class Net:                     # float IR under construction
    def __init__(s, h, w, c): s.t = [Tn(h, w, c, True)]; s.ops = []; s.name = ''
    def tensor(s, h, w, c, signed): s.t.append(Tn(h, w, c, signed)); return len(s.t) - 1
    def conv(s, src, W, b, stride, pad, groups=1, relu=True, post=None, pool=(1, 1), tag=''):
        """W: (cout, kh, kw, cin/groups) float; post = (g, bb) per-channel affine applied after relu/pool."""
        ti = s.t[src]; cout, kh, kw, _ = W.shape
        oh = (ti.h + 2 * pad[0] - kh) // stride[0] + 1; ow = (ti.w + 2 * pad[1] - kw) // stride[1] + 1
        oh, ow = oh // pool[0], ow // pool[1]
        out = s.tensor(oh, ow, cout, signed=(not relu) or post is not None)
        s.ops.append(dict(type=OP_CONV, a=src, out=out, W=W, b=b, stride=stride, pad=pad, groups=groups, relu=relu, post=post,
                          pool=pool, tag=tag))
        return out
    def fc(s, src, W, b, relu=True, tag=''):
        out = s.tensor(1, 1, W.shape[0], signed=not relu)
        s.ops.append(dict(type=OP_FC, a=src, out=out, W=W, b=b, relu=relu, tag=tag)); return out
    def add(s, a, b, relu, tag=''):
        ta = s.t[a]; out = s.tensor(ta.h, ta.w, ta.c, signed=not relu)
        s.ops.append(dict(type=OP_ADD, a=a, bt=b, out=out, relu=relu, tag=tag)); return out
    def affine(s, src, g, bb, tag=''):
        ta = s.t[src]; out = s.tensor(ta.h, ta.w, ta.c, True)
        s.ops.append(dict(type=OP_AFFINE, a=src, out=out, g=g, bb=bb, tag=tag)); return out
    def gapfc(s, src, W, b, tag='head'):
        out = s.tensor(1, 1, W.shape[0], True)
        s.ops.append(dict(type=OP_GAPFC, a=src, out=out, W=W, b=b, tag=tag)); return out


def _bn(bn):                   # eval-mode BatchNorm -> per-channel (g, shift)
    g = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    return g.numpy(), (bn.bias.detach().double() - bn.running_mean.double() * g).numpy()


def _fold_conv(conv, bn):      # Conv(+BN) -> OHWI weights + bias
    W = conv.weight.detach().double()
    if W.dim() == 3: W = W.unsqueeze(2)                                   # Conv1d (o,i,k) -> (o,i,1,k)
    if bn is not None:
        g, sh = _bn(bn); W = W * torch.from_numpy(g).view(-1, 1, 1, 1); b = sh
    else: b = np.zeros(W.shape[0]) if conv.bias is None else conv.bias.detach().double().numpy()
    return W.permute(0, 2, 3, 1).contiguous().numpy(), b                  # (o, kh, kw, i)


def build_float_ir(key, model, kind, shape):
    from common.zoo import DNN, TCResNet, DSCNN, ResKWS
    T, Fm = shape
    net = Net(T, Fm, 1); net.name = key
    if isinstance(model, DNN):
        mods = list(model.net); x = 0; i = 0
        lins = [m for m in mods if isinstance(m, nn.Linear)]; bns = [m for m in mods if isinstance(m, nn.BatchNorm1d)]
        for L, B in zip(lins[:-1], bns):
            g, sh = _bn(B); W = L.weight.detach().double().numpy() * g[:, None]
            x = net.fc(x, W, sh, relu=True, tag='fc%d' % i); i += 1
        L = lins[-1]; net.gapfc(x, L.weight.detach().double().numpy(), L.bias.detach().double().numpy(), tag='logits')
    elif isinstance(model, TCResNet):
        W, b = _fold_conv(model.conv, None); net.t[0].h, net.t[0].w, net.t[0].c = 1, T, Fm
        x = net.conv(0, W, b, (1, 1), (0, 1), relu=False, tag='conv0')
        for bi, blk in enumerate(model.blocks):
            st = blk.c1.stride[0]
            if blk.sc is None: r = x
            else:
                Ws, bs = _fold_conv(blk.sc[0], blk.sc[1]); r = net.conv(x, Ws, bs, (1, st), (0, 0), relu=True, tag='b%d.sc' % bi)
            W1, b1 = _fold_conv(blk.c1, blk.b1); a = net.conv(x, W1, b1, (1, st), (0, 4), relu=True, tag='b%d.c1' % bi)
            W2, b2 = _fold_conv(blk.c2, blk.b2); y = net.conv(a, W2, b2, (1, 1), (0, 4), relu=False, tag='b%d.c2' % bi)
            x = net.add(r, y, relu=True, tag='b%d.add' % bi)
        net.gapfc(x, model.fc.weight.detach().double().numpy(), model.fc.bias.detach().double().numpy())
    elif isinstance(model, DSCNN):
        W, b = _fold_conv(model.conv, model.bn); x = net.conv(0, W, b, (2, 2), (5, 1), relu=True, tag='conv0')
        for bi, blk in enumerate(model.blocks):
            Wd, bd = _fold_conv(blk.dw, blk.b1)                                   # (c,3,3,1)
            x = net.conv(x, Wd, bd, (1, 1), (1, 1), groups=Wd.shape[0], relu=True, tag='dw%d' % bi)
            Wp, bp = _fold_conv(blk.pw, blk.b2); x = net.conv(x, Wp, bp, (1, 1), (0, 0), relu=True, tag='pw%d' % bi)
        net.gapfc(x, model.fc.weight.detach().double().numpy(), model.fc.bias.detach().double().numpy())
    elif isinstance(model, ResKWS):
        W, b = _fold_conv(model.conv0, None); pool = model.pool.kernel_size if isinstance(model.pool.kernel_size, tuple) else (model.pool.kernel_size,) * 2
        P = net.conv(0, W, b, (1, 1), (1, 1), relu=True, pool=tuple(pool), tag='conv0+pool'); x = P; old = P
        for i in range(1, model.n + 1):
            Wc, bc = _fold_conv(model.convs[i - 1], None); g, sh = _bn(model.bns[i - 1])
            if i % 2 == 1: x = net.conv(x, Wc, bc, (1, 1), (1, 1), relu=True, post=(g, sh), tag='conv%d' % i)
            else:
                Y = net.conv(x, Wc, bc, (1, 1), (1, 1), relu=True, tag='conv%d' % i)
                S = net.add(Y, old, relu=True, tag='res%d' % i); old = S; x = net.affine(S, g, sh, tag='bn%d' % i)
        net.gapfc(x, model.fc.weight.detach().double().numpy(), model.fc.bias.detach().double().numpy())
    else: raise ValueError(key)
    return net


# ----------------------------------------------------------------------------------------------------------------
# float executor (torch, batched) -> used for calibration and to verify the IR against the original nn.Module
def run_float(net, X, collect=False):
    """X: N x T x F float32 -> logits N x 12 (and per-tensor arrays if collect)."""
    ts = {}; t0 = net.t[0]
    ts[0] = torch.from_numpy(np.asarray(X, np.float64)).reshape(len(X), t0.h, t0.w, t0.c).permute(0, 3, 1, 2)   # NCHW
    for op in net.ops:
        ty = op['type']
        if ty == OP_CONV:
            W = torch.from_numpy(op['W']).permute(0, 3, 1, 2); y = F.conv2d(ts[op['a']], W, torch.from_numpy(op['b']), op['stride'], op['pad'], groups=op['groups'])
            if op['relu']: y = F.relu(y)
            if op['pool'] != (1, 1): y = F.avg_pool2d(y, op['pool'])
            if op['post'] is not None: g, bb = op['post']; y = y * torch.from_numpy(g).view(1, -1, 1, 1) + torch.from_numpy(bb).view(1, -1, 1, 1)
        elif ty == OP_FC:
            x = ts[op['a']].permute(0, 2, 3, 1).reshape(len(X), -1); y = x @ torch.from_numpy(op['W']).T + torch.from_numpy(op['b'])
            if op['relu']: y = F.relu(y)
            y = y.view(len(X), -1, 1, 1)
        elif ty == OP_ADD:
            y = ts[op['a']] + ts[op['bt']]
            if op['relu']: y = F.relu(y)
        elif ty == OP_AFFINE:
            y = ts[op['a']] * torch.from_numpy(op['g']).view(1, -1, 1, 1) + torch.from_numpy(op['bb']).view(1, -1, 1, 1)
        elif ty == OP_GAPFC:
            x = ts[op['a']].mean((2, 3)); y = (x @ torch.from_numpy(op['W']).T + torch.from_numpy(op['b'])).view(len(X), -1, 1, 1)
        ts[op['out']] = y
    logits = ts[net.ops[-1]['out']].reshape(len(X), -1).numpy()
    return (logits, ts) if collect else logits


def calibrate(net, Xcal, pct=None, bs=200, nsamp=40000):
    """Choose each tensor's clipping range over the calibration clips and set its scale (int8: /127, uint8: /255).
    pct=None: pick the clip that minimises the quantisation MSE of the tensor (robust to heavy tails);
    pct=float: fixed |value| percentile."""
    rng = np.random.RandomState(0); samp = {i: [] for i in range(len(net.t))}
    for i in range(0, len(Xcal), bs):
        _, ts = run_float(net, Xcal[i:i + bs], collect=True)
        for k, v in ts.items():
            a = v.numpy().ravel(); samp[k].append(a[rng.randint(0, len(a), min(nsamp, len(a)))])
    for k, t in enumerate(net.t):
        v = np.concatenate(samp[k]); qmax = 127.0 if t.signed else 255.0; av = np.abs(v)
        if pct is not None: clip = np.percentile(av, pct)
        else:
            best = None
            for c in np.concatenate([np.percentile(av, [99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]), av.max() * np.linspace(0.3, 1.0, 15)]):
                sc = max(c, 1e-8) / qmax; q = np.clip(np.round(v / sc), -127 if t.signed else 0, qmax) * sc
                e = float(((v - q) ** 2).mean())
                if best is None or e < best[0]: best = (e, c)
            clip = best[1]
        t.scale = max(float(clip), 1e-8) / qmax


# ----------------------------------------------------------------------------------------------------------------
def _mult_shift(r):
    """r (float, may be negative) -> (int32 mult, shift) with value ~= mult * 2^-shift."""
    mt, e = math.frexp(abs(r)); mu = int(round(mt * 2 ** 31)); sh = 31 - e
    if mu == 2 ** 31: mu //= 2; sh -= 1
    assert 1 <= sh < 62, (r, sh)
    return (mu if r >= 0 else -mu), sh


def _mult_shift_vec(rv):
    m = np.zeros(len(rv), np.int64); s = np.zeros(len(rv), np.int64)
    for i, r in enumerate(rv): m[i], s[i] = _mult_shift(float(r)) if r != 0 else (0, 31)
    return m, s


def quantize(net):
    """Float IR (+ calibrated tensor scales) -> list of integer ops."""
    q = []
    for op in net.ops:
        ty = op['type']; ti = net.t[op['a']]; to = net.t[op['out']]; d = dict(type=ty, a=op['a'], out=op['out'], tag=op['tag'])
        if ty in (OP_CONV, OP_FC, OP_GAPFC):
            W = op['W']; b = op['b']
            axes = tuple(range(1, W.ndim)); sw = np.abs(W).max(axis=axes) / 127.0; sw = np.maximum(sw, 1e-12)
            Wq = np.clip(np.round(W / sw.reshape((-1,) + (1,) * (W.ndim - 1))), -127, 127).astype(np.int8)
            if ty == OP_CONV and op['groups'] > 1: Wq = np.ascontiguousarray(Wq[:, :, :, 0].transpose(1, 2, 0))   # depthwise: (kh,kw,c)
            d['Wq'] = Wq
            if ty == OP_GAPFC:
                d['fscale'] = (sw * ti.scale / (ti.h * ti.w)).astype(np.float32); d['fbias'] = b.astype(np.float32); d['sw'] = sw
            else:
                sacc = ti.scale * sw; d['Bq'] = np.round(b / sacc).astype(np.int64)
                assert np.abs(d['Bq']).max() < 2 ** 31
                if ty == OP_CONV and op['post'] is not None:
                    g, bb = op['post']; r = g * sacc / to.scale; d['boff'] = np.round(bb / to.scale).astype(np.int64)
                else: r = sacc / to.scale; d['boff'] = None
                d['M'], d['SH'] = _mult_shift_vec(r); d['relu'] = op['relu']
            d.update(groups=op.get('groups', 1), stride=op.get('stride', (1, 1)), pad=op.get('pad', (0, 0)), pool=op.get('pool', (1, 1)))
            d['kh'], d['kw'] = (W.shape[1], W.shape[2]) if ty == OP_CONV else (1, 1)
        elif ty == OP_ADD:
            tb = net.t[op['bt']]; ra, rb = ti.scale / to.scale, tb.scale / to.scale
            sh = 30 - int(math.ceil(math.log2(max(ra, rb, 1e-9)))); sh = min(max(sh, 8), 40)
            d.update(bt=op['bt'], ma=int(round(ra * 2 ** sh)), mb=int(round(rb * 2 ** sh)), ash=sh, relu=op['relu'])
        elif ty == OP_AFFINE:
            r = op['g'] * ti.scale / to.scale; d['M'], d['SH'] = _mult_shift_vec(r); d['boff'] = np.round(op['bb'] / to.scale).astype(np.int64)
        q.append(d)
    return q


# ----------------------------------------------------------------------------------------------------------------
def _epilogue(acc, M, SH, boff, relu, signed_out, hi=None):
    """acc int64 (..., C) -> clipped int64 outputs. Identical to the device code."""
    if relu: acc = np.maximum(acc, 0)
    assert np.abs(acc).max(initial=0) * int(np.abs(M).max()) < 2 ** 62, 'requant overflow'
    v = (acc * M + (np.int64(1) << (SH - 1))) >> SH
    if boff is not None: v = v + boff
    hi = hi or (127 if signed_out else 255)                  # low-bit variants (lowbit.py) narrow the output range per op
    return np.clip(v, -hi if signed_out else 0, hi)


def _windows(x, kh, kw, sh, sw, ph, pw):                # x: N,H,W,C -> N,oh,ow,kh,kw,C (zero padded)
    x = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))      # N,H',W',C,kh,kw
    return win[:, ::sh, ::sw].transpose(0, 1, 2, 4, 5, 3)


def run_int(net, qops, xq, return_all=False):
    """xq: N x h x w x c int (already quantised input). Returns (int32 accumulators N x ncls, float32 logits)."""
    ts = {0: xq.astype(np.int64)}
    for d in qops:
        ty = d['type']; to = net.t[d['out']]
        if ty == OP_CONV:
            x = ts[d['a']]; N = x.shape[0]; kh, kw = d['kh'], d['kw']
            win = _windows(x.astype(np.float64), kh, kw, *d['stride'], *d['pad'])            # N,oh,ow,kh,kw,C
            oh, ow = win.shape[1:3]
            if d['groups'] > 1:
                acc = np.einsum('nhwijc,ijc->nhwc', win, d['Wq'].astype(np.float64))
            else:
                cout = d['Wq'].shape[0]
                acc = win.reshape(N * oh * ow, -1) @ d['Wq'].reshape(cout, -1).astype(np.float64).T
                acc = acc.reshape(N, oh, ow, cout)
            acc = np.rint(acc).astype(np.int64) + d['Bq']
            ph, pw_ = d['pool']
            if (ph, pw_) != (1, 1):
                y = _epilogue(acc, d['M'], d['SH'], None, d['relu'], False, d.get('out_hi'))
                H, W_ = (oh // ph) * ph, (ow // pw_) * pw_
                s = y[:, :H, :W_].reshape(N, H // ph, ph, W_ // pw_, pw_, -1).sum((2, 4)); n = ph * pw_
                y = (s + n // 2) // n
            else: y = _epilogue(acc, d['M'], d['SH'], d['boff'], d['relu'], to.signed, d.get('out_hi'))
            ts[d['out']] = y
        elif ty == OP_FC:
            x = ts[d['a']].reshape(len(ts[d['a']]), -1)
            acc = np.rint(x.astype(np.float64) @ d['Wq'].astype(np.float64).T).astype(np.int64) + d['Bq']
            ts[d['out']] = _epilogue(acc, d['M'], d['SH'], None, d['relu'], to.signed, d.get('out_hi')).reshape(len(x), 1, 1, -1)
        elif ty == OP_ADD:
            a, b = ts[d['a']], ts[d['bt']]
            v = (a * d['ma'] + b * d['mb'] + (np.int64(1) << (d['ash'] - 1))) >> d['ash']
            hi = d.get('out_hi') or (127 if to.signed else 255)
            ts[d['out']] = np.clip(np.maximum(v, 0) if d['relu'] else v, -hi if to.signed else 0, hi)
        elif ty == OP_AFFINE:
            ts[d['out']] = _epilogue(ts[d['a']], d['M'], d['SH'], d['boff'], False, True, d.get('out_hi'))
        elif ty == OP_GAPFC:
            x = ts[d['a']]; S = x.reshape(len(x), -1, x.shape[-1]).sum(1)
            acc = np.rint(S.astype(np.float64) @ d['Wq'].astype(np.float64).T).astype(np.int64)
            assert np.abs(acc).max() < 2 ** 31
            logits = (acc.astype(np.float32) * d['fscale'] + d['fbias']).astype(np.float32)
            return (acc.astype(np.int32), logits, ts) if return_all else (acc.astype(np.int32), logits)
    raise RuntimeError('no head')


# ----------------------------------------------------------------------------------------------------------------
def plan_arena(net):
    """Linear-scan allocation of activation tensors (live from producer to last consumer). Returns offsets, peak bytes."""
    last = {}
    for i, op in enumerate(net.ops):
        for k in ('a', 'bt'):
            if k in op: last[op[k]] = i
    last[0] = max(last.get(0, 0), 0)
    off = {}; peak = 0
    # tensor 0 is the externally written input: lives in the arena too
    cur = []   # (offset, end_off, last_use_op, tid)
    order = [0] + [op['out'] for op in net.ops]
    def first_free(size, start_op):
        cur[:] = [c for c in cur if c[2] >= start_op]
        pos = 0
        for o, e, _, _ in sorted(cur):
            if pos + size <= o: break
            pos = max(pos, e)
        return pos
    for idx, tid in enumerate(order):
        start_op = idx - 1 if idx > 0 else -1          # op index that produces it
        size = (net.t[tid].n + 3) & ~3
        o = first_free(size, start_op)                 # tensors whose last use is >= producing op stay live (in/out can't overlap)
        off[tid] = o; cur.append((o, o + size, last.get(tid, len(net.ops)), tid)); peak = max(peak, o + size)
    return off, peak


def _macs(net, qops):
    tot = [];
    for d in qops:
        ty = d['type']; to = net.t[d['out']]; ti = net.t[d['a']]
        if ty == OP_CONV:
            cin_g = ti.c // d['groups']
            ph, pw_ = d['pool']; oh, ow = (to.h * ph, to.w * pw_) if ph * pw_ > 1 else (to.h, to.w)
            K = d['kh'] * d['kw'] * cin_g; n = oh * ow * to.c * K
        elif ty == OP_FC: K = ti.n; n = to.c * K
        elif ty == OP_GAPFC: K = ti.c; n = d['Wq'].shape[0] * K
        else: K = 0; n = 0
        tot.append((n, K))
    return tot


def write_blob(net, qops, path, name, s_in):
    off, peak = plan_arena(net)
    data = bytearray()
    def put(arr, dt):
        nonlocal data
        while len(data) % 4: data.append(0)
        o = len(data); data += np.ascontiguousarray(arr, dtype=dt).tobytes(); return o
    ops_b = bytearray(); macs = _macs(net, qops)
    for d, (nm, K) in zip(qops, macs):
        ty = d['type']; ti = net.t[d['a']]; to = net.t[d['out']]
        r = dict(type=ty, relu=int(d.get('relu', False)), in_signed=int(ti.signed), out_signed=int(to.signed),
                 in_signed_b=int(net.t[d['bt']].signed) if ty == OP_ADD else 0, flags=0,
                 r0=d.get('out_hi', 0), r1=d.get('wmax', 0), pad=d.get('in_max', 0),   # low-bit variants only; 0 = int8 default
                 a=d['a'], b=d['bt'] if ty == OP_ADD else -1, out=d['out'], out2=-1,
                 cin=ti.c, cout=to.c, groups=d.get('groups', 1), kh=d.get('kh', 1), kw=d.get('kw', 1),
                 sh=d.get('stride', (1, 1))[0], sw=d.get('stride', (1, 1))[1], ph=d.get('pad', (0, 0))[0], pw=d.get('pad', (0, 0))[1],
                 pool_h=d.get('pool', (1, 1))[0], pool_w=d.get('pool', (1, 1))[1], ih=ti.h, iw=ti.w, oh=to.h, ow=to.w,
                 w_off=0, b_off=0, m_off=0, sh_off=0, bo_off=0, fs_off=0, fb_off=0, ma=d.get('ma', 0), mb=d.get('mb', 0), ash=d.get('ash', 0),
                 k=K, macs=nm)
        if ty == OP_CONV and d['groups'] > 1: r['flags'] = 1                              # depthwise
        if ty == OP_CONV and d['pool'] != (1, 1): r['oh'], r['ow'] = to.h * d['pool'][0], to.w * d['pool'][1]   # pre-pool conv grid
        if ty == OP_FC: r['cin'] = ti.n
        if 'Wq' in d: r['w_off'] = put(d['Wq'], np.int8)
        if 'Bq' in d: r['b_off'] = put(d['Bq'], np.int32)
        if 'M' in d: r['m_off'] = put(d['M'], np.int32); r['sh_off'] = put(d['SH'], np.uint8)
        if d.get('boff') is not None: r['bo_off'] = put(d['boff'], np.int16)
        if 'fscale' in d: r['fs_off'] = put(d['fscale'], np.float32); r['fb_off'] = put(d['fbias'], np.float32)
        ops_b += struct.pack('<8B 4h 4h 4h 4h 4h 7I iiii I I 28x',
                             r['type'], r['relu'], r['in_signed'], r['out_signed'], r['in_signed_b'], r['flags'], r['r0'], r['r1'],
                             r['a'], r['b'], r['out'], r['out2'], r['cin'], r['cout'], r['groups'], r['pad'],
                             r['kh'], r['kw'], r['sh'], r['sw'], r['ph'], r['pw'], r['pool_h'], r['pool_w'], r['ih'], r['iw'], r['oh'], r['ow'],
                             r['w_off'], r['b_off'], r['m_off'], r['sh_off'], r['bo_off'], r['fs_off'], r['fb_off'],
                             r['ma'], r['mb'], r['ash'], 0, r['k'], min(r['macs'], 2 ** 32 - 1))
    assert len(ops_b) % len(qops) == 0 and len(ops_b) // len(qops) == 128, len(ops_b) // len(qops)
    ten_b = b''.join(struct.pack('<IIhhhBx', off[i], t.n, t.h, t.w, t.c, int(t.signed)) for i, t in enumerate(net.t))
    hdr_size = 128; ops_off = hdr_size; ten_off = ops_off + len(ops_b); data_off = (ten_off + len(ten_b) + 15) & ~15
    total_macs = sum(m for m, _ in macs); params = sum(d['Wq'].size for d in qops if 'Wq' in d)
    hdr = struct.pack('<4s I 16s 8I f 6I', b'KWSM', 1, name.encode(), len(qops), len(net.t), ops_off, ten_off, data_off, peak, net.t[0].h * 0 + net.t[0].n, len(qops) - 1,
                      s_in, net.t[0].h, net.t[0].w, net.t[0].c, int(net.t[qops[-1]['out']].c), total_macs & 0xffffffff, params)
    hdr = hdr.ljust(hdr_size, b'\0')
    blob = hdr + bytes(ops_b) + ten_b
    blob = blob.ljust(data_off, b'\0') + bytes(data)
    with open(path, 'wb') as f: f.write(blob)
    return dict(arena_bytes=peak, blob_bytes=len(blob), macs=total_macs, weights=params, n_ops=len(qops))
