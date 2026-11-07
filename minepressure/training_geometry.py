"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np

def build_source_geometry_and_pairs(meta, source, truth, known, eligible):
    source = np.asarray(source, dtype=np.int64)
    m = meta.iloc[source]
    if truth.shape != (len(source), 8, 10) or known.shape != truth.shape or eligible.shape != (len(source), 10):
        raise ValueError('Source-only physical target arrays have inconsistent shape')
    if (known & (~np.isfinite(truth) | (truth < 0))).any():
        raise ValueError('Invalid known physical value')
    face = m.face_index.to_numpy()
    issue = m.issue_index.to_numpy()
    keys = np.column_stack((np.repeat(face, 8), (issue[:, None] + np.arange(8)).reshape(-1)))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    values = np.full((len(unique), 10), np.nan, np.float64)
    yy = truth.reshape(-1, 10)
    kk = known.reshape(-1, 10)
    for j in range(10):
        lo = np.full(len(unique), np.inf)
        hi = np.full(len(unique), -np.inf)
        np.minimum.at(lo, inverse[kk[:, j]], yy[kk[:, j], j])
        np.maximum.at(hi, inverse[kk[:, j]], yy[kk[:, j], j])
        good = np.isfinite(lo)
        if not np.array_equal(lo[good], hi[good]):
            raise ValueError('Repeated source target observations disagree')
        values[good, j] = lo[good]
    edges = []
    for j in range(10):
        normal = values[:, j][np.isfinite(values[:, j]) & (values[:, j] <= 1)] - 1
        if len(normal) < 8:
            raise ValueError('Insufficient source normal geometry support')
        edges.append(np.r_[-1.0, np.quantile(normal, np.arange(1, 8) / 8), 0.0])
    edges = np.asarray(edges, dtype=np.float32)
    if not (np.diff(edges, axis=-1) > 0).all():
        raise ValueError('Normal quantiles collapse; resolve state capacity before fitting')
    gap = values.astype(np.float32) - np.float32(1)
    state = (gap[..., None] > edges[None, :, 1:-1]).sum(-1).astype(np.int64)
    state = np.where(values > 1, 8, state)
    state = np.where(np.isfinite(values), state, -1)
    lookup = {tuple(key): i for i, key in enumerate(unique.tolist())}
    previous = np.full((len(source), 9, 10), -1, np.int64)
    following = np.full_like(previous, -1)
    for n, (f, issue_index) in enumerate(zip(face, issue)):
        for h in range(9):
            left = lookup.get((int(f), int(issue_index) - 2 + h))
            right = lookup.get((int(f), int(issue_index) - 1 + h))
            if left is not None:
                previous[n, h] = state[left]
            if right is not None:
                following[n, h] = state[right]
    pair_known = (previous >= 0) & (following >= 0) & eligible[:, None, :]
    previous = np.where(pair_known, previous, -1)
    following = np.where(pair_known, following, -1)
    return dict(normal_edges=edges, unique_point_keys=unique, unique_source_value=values, previous_state=previous, next_state=following, pair_known=pair_known, source_origins=source.copy())
