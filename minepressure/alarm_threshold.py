"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np

def f_cut(q, y, k, beta):
    v = q[k]
    t = y[k]
    assert np.isfinite(v).all() and t.any()
    order = np.argsort(-v, kind='stable')
    v = v[order]
    t = t[order]
    ends = np.r_[np.flatnonzero(v[:-1] != v[1:]), len(v) - 1]
    tp = t.cumsum()[ends]
    fp = (~t).cumsum()[ends]
    b = beta * beta
    f = (1 + b) * tp / (b * t.sum() + tp + fp)
    ix = np.flatnonzero(f == f.max())
    best = ix[np.argmin(fp[ix])]
    return float(v[ends[best]])
