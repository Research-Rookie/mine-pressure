"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np
from sklearn.cluster import KMeans

def weighted_quantile(x, w, q):
    x = np.asarray(x, np.float64)
    w = np.asarray(w, np.float64)
    order = np.argsort(x, kind='stable')
    cdf = np.cumsum(w[order])
    assert cdf[-1] > 0
    return float(x[order[min(np.searchsorted(cdf, q * cdf[-1], side='left'), len(x) - 1)]])

def initialize(selected, event_keys, mean, scale, count):
    x = np.asarray(selected, np.float64)
    keys = np.asarray(event_keys, np.int64)
    mean = np.asarray(mean, np.float64)
    scale = np.asarray(scale, np.float64)
    assert x.ndim == 2 and x.shape[1] == 6 and (len(x) == len(keys)) and np.isfinite(x).all()
    events, inverse, sizes = np.unique(keys, return_inverse=True, return_counts=True)
    assert len(events) >= count, 'Insufficient FIT event representatives;stop instead of changing counts.'
    z = (x - mean) / scale
    representative = np.stack([np.median(z[inverse == j], axis=0) for j in range(len(events))])
    centre = np.median(representative, axis=0)
    first = int(np.argmin(np.square(representative - centre).sum(1)))
    chosen = [first]
    nearest = np.square(representative - representative[first]).sum(1)
    for _ in range(1, count):
        nearest[chosen] = -np.inf
        idx = int(np.argmax(nearest))
        assert nearest[idx] > 1e-12, 'Duplicate event medians;stop.'
        chosen.append(idx)
        nearest = np.minimum(nearest, np.square(representative - representative[idx]).sum(1))
    weights = 1 / sizes[inverse].astype(np.float64)
    cluster = KMeans(n_clusters=count, init=representative[chosen].copy(), n_init=1, random_state=0, algorithm='lloyd', max_iter=300, tol=0.0001).fit(z, sample_weight=weights)
    widths = []
    support = []
    for k in range(count):
        use = cluster.labels_ == k
        assert use.any()
        delta = np.abs(z[use] - cluster.cluster_centers_[k])
        widths.append([max(0.05, weighted_quantile(delta[:, d], weights[use], 0.9)) for d in range(6)])
        support.append(int(len(np.unique(keys[use]))))
    state = dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32), center=cluster.cluster_centers_.astype(np.float32), width=np.asarray(widths, np.float32), canonical_center=np.zeros((count, 6), np.float32), canonical_scale=np.ones(6, np.float32))
    proof = dict(events=len(events), positive_source_cells=len(x), rules=count, initial_representative_event_keys=events[chosen].tolist(), events_per_final_rule=support, max_equal_event_weight_error=float(np.max(np.abs(np.bincount(inverse, weights=weights) - 1))), initial_method='Median event representative nearest globalmedian,then farthest-first with sorted-event tie breaks;weightedLloyd on all positiveFIT cells;eventweighted90percentile width floor.05.', future_magnitude_input=False, random_restart=False, new_trainable_parameters=0)
    return (state, proof)
