"""Blob parser and input quantiser.

load_blob is copied from cryptonite benchmark/export/lowbit.py (the original also pulls in the model zoo and
firmware paths, which are not needed here); quant_input is from the same file.

One change to load_blob: the original ignores the per-op output clip that the low-bit variants store in the first
spare OpRec byte (r0, see write_blob in qmodel.py), so a w4a4 blob loaded with it runs with 8-bit activation ranges and
gives ~96% prediction agreement with the stored w4a4 simulation. Here r0 is read back into d['out_hi'], which run_int
already understands (0 means the int8 default).
"""
import struct
import numpy as np
import qmodel as Q


class Net:
    def __init__(s, tens, ops): s.t = tens; s.ops = ops


def load_blob(path):
    b = open(path, 'rb').read()
    magic, ver, name, n_ops, n_ten, ops_off, ten_off, data_off, arena, in_n, head, s_in, ih, iw, ic, ncls, macs, params = struct.unpack_from('<4s I 16s 8I f 6I', b, 0)
    assert magic == b'KWSM'
    tens = []
    for i in range(n_ten):
        off, n, h, w, c, sg = struct.unpack_from('<IIhhhBx', b, ten_off + 16 * i); t = Q.Tn(h, w, c, bool(sg)); tens.append(t)
    def arr(dt, off, n): return np.frombuffer(b, dtype=dt, count=n, offset=data_off + off).copy()
    qops = []
    for i in range(n_ops):
        f = struct.unpack_from('<8B 4h 4h 4h 4h 4h 7I iiii I I', b, ops_off + 128 * i)
        ty, relu, in_s, out_s, in_sb, flags, r0 = f[:7]; a, bt, out, out2, cin, cout, groups, _pad, kh, kw, sh, sw, ph, pw, pool_h, pool_w, ih_, iw_, oh_, ow_ = f[8:28]
        w_off, b_off, m_off, sh_off, bo_off, fs_off, fb_off, ma, mb, ash, _p2, k, _m = f[28:41]
        d = dict(type=ty, a=a, out=out, tag='op%d' % i)
        if r0: d['out_hi'] = r0
        if ty in (Q.OP_CONV, Q.OP_FC):
            if ty == Q.OP_CONV and flags & 1: d['Wq'] = arr(np.int8, w_off, kh * kw * cin).reshape(kh, kw, cin)
            elif ty == Q.OP_CONV: d['Wq'] = arr(np.int8, w_off, cout * kh * kw * (cin // groups)).reshape(cout, kh, kw, cin // groups)
            else: d['Wq'] = arr(np.int8, w_off, cout * cin).reshape(cout, cin)
            d['Bq'] = arr(np.int32, b_off, cout).astype(np.int64); d['M'] = arr(np.int32, m_off, cout).astype(np.int64); d['SH'] = arr(np.uint8, sh_off, cout).astype(np.int64)
            d['boff'] = arr(np.int16, bo_off, cout).astype(np.int64) if bo_off else None
            d['relu'] = bool(relu)
            if ty == Q.OP_CONV: d.update(groups=groups, stride=(sh, sw), pad=(ph, pw), pool=(pool_h, pool_w), kh=kh, kw=kw)
            else: d.update(groups=1, stride=(1, 1), pad=(0, 0), pool=(1, 1), kh=1, kw=1)
        elif ty == Q.OP_ADD: d.update(bt=bt, ma=ma, mb=mb, ash=ash, relu=bool(relu))
        elif ty == Q.OP_AFFINE:
            d['M'] = arr(np.int32, m_off, cout).astype(np.int64); d['SH'] = arr(np.uint8, sh_off, cout).astype(np.int64); d['boff'] = arr(np.int16, bo_off, cout).astype(np.int64)
        elif ty == Q.OP_GAPFC:
            d['Wq'] = arr(np.int8, w_off, cout * cin).reshape(cout, cin); d['fscale'] = arr(np.float32, fs_off, cout); d['fbias'] = arr(np.float32, fb_off, cout)
        qops.append(d)
    ops = [dict(a=d['a'], out=d['out'], **({'bt': d['bt']} if 'bt' in d else {})) for d in qops]
    return Net(tens, ops), qops, s_in, name.rstrip(b'\0').decode()


def quant_input(X, s_in, net, hi0):
    t = net.t[0]; return np.clip(np.round(X / s_in), -hi0, hi0).astype(np.int64).reshape(len(X), t.h, t.w, t.c)
