import numpy as np
import torch
from scipy.special import expit

@torch.no_grad()
def predict_arrays(bundle, sequence, summary_raw, anchor_raw, installed):
    from .four_parameter_conditions import four_features
    from .conditional_normal_inputs import frozen_body_inputs
    from .regional_maximum_distribution import maximum_atomic_distribution
    from .tail_preserving_normal_projection import numpy_projection
    device = bundle['device']
    n = len(sequence)
    sequence = np.asarray(sequence, np.float32).copy()
    summary_raw = np.asarray(summary_raw, np.float32)
    anchor_raw = np.asarray(anchor_raw, np.float32)
    installed = np.asarray(installed, bool)
    raw_features = bundle['preprocessing']['imputer'].n_features_in_
    assert sequence.shape == (n, 6, 24, 28) and summary_raw.shape == (n, 6, raw_features) and (anchor_raw.shape == (n, 6, 11)) and (installed.shape == (n, 6, 11))
    sequence[..., 11:22] *= installed[:, :, None, :]
    limits = np.asarray(bundle['engineering_limits'], np.float32)
    anchor = np.where(installed, anchor_raw / limits, np.nan).astype(np.float32)
    support = np.isfinite(summary_raw[..., :11]) & installed
    pre = bundle['preprocessing']
    summary = pre['scaler'].transform(pre['imputer'].transform(summary_raw.reshape(-1, raw_features))).astype(np.float32).reshape(n, 6, 277)
    data = dict(oi=np.arange(n), seq=sequence, anchor=anchor, support=support, installed=installed)
    numeric = bundle['numeric'](*[torch.as_tensor(v, device=device) for v in [sequence, summary, anchor, support]]).cpu().numpy()
    cache = frozen_body_inputs(bundle['parent'], data, device=device, batch=bundle['body_batch_size'])
    if not cache['eligible'].any(1).all():
        raise ValueError('At least one eligible region per physical parameter required by this frozen-model input interface.')
    features = four_features(data, device, np.asarray(bundle['lower_limits']) / limits,
                             bundle['body_batch_size'])
    cols = []
    for a, model in enumerate(bundle['warning']):
        cols.append(model(torch.as_tensor(features[..., a, :], device=device)).cpu().numpy())
    regional_score = np.stack(cols, -1)
    raw = np.where(support[..., [0, 1, 9, 10]][:, :, None, :], regional_score, -np.inf).max(1)
    links = np.array(bundle['probability_links'])
    ep = cache['event_probability'].astype(np.float64)
    for a, j in enumerate([0, 1, 8, 9]):
        ep[:, :, j, :] = expit(links[a, 0] * regional_score[..., a] + links[a, 1])
    cache['event_probability'] = ep
    out = bundle['body'](*[torch.as_tensor(cache[k], device=device) for k in ['own', 'dynamics', 'initial_normal', 'event_probability', 'positive_atom']])
    p = out['probability'].double()
    atoms = out['atoms'].double()
    pp = p.permute(0, 2, 3, 1, 4)
    active = torch.as_tensor(cache['eligible'], device=device).transpose(1, 2)[:, :, None, :].expand(*pp.shape[:-1])
    law = maximum_atomic_distribution(pp, atoms.permute(0, 2, 3, 1, 4), active, torch.ones(pp.shape[:-2], device=device, dtype=torch.float64))
    w = law['probability'].transpose(1, 2).cpu().numpy()
    a = law['atoms'].transpose(1, 2).cpu().numpy()
    wanted = np.where(cache['eligible'][:, :, None, :], numeric, -np.inf).max(1)
    proj = numpy_projection(w, a, wanted)
    c = torch.as_tensor(proj['normal_scale'], device=device).transpose(1, 2)[:, None, :, :, None]
    b = torch.as_tensor(proj['normal_bias'], device=device).transpose(1, 2)[:, None, :, :, None]
    mapped = torch.where(atoms <= 0, (c * (1 + atoms) + b - 1).clamp(-1, 0), atoms)
    cuts = np.array(bundle['raw_cuts'])
    risk = (w * (a > 0)).sum(-1)
    return dict(face_probability=w, face_atoms=proj['gaps'], face_value=proj['mean'], face_risk=risk, raw_face_score=raw, native_flags=raw >= cuts, regional_value=(p * (1 + mapped)).sum(-1).transpose(-1, -2).cpu().numpy(), requested_mean=wanted, probability_cut=expit(links[:, 0] * cuts + links[:, 1]), eligible=cache['eligible'])
