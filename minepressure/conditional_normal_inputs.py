"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np
import torch
from .asymmetric_boundary_flux_network import PHYSICAL
from .boundary_observation_joint import joint_forward

def masked_slope(values, observed, window):
    v = values[:, :, -window:]
    m = observed[:, :, -window:]
    time = torch.arange(-window + 1, 1, device=v.device, dtype=v.dtype)[None, None, :, None]
    count = m.sum(2)
    mean_t = (m * time).sum(2) / count.clamp_min(1)
    mean_v = (m * v).sum(2) / count.clamp_min(1)
    dt = time - mean_t[:, :, None]
    dv = v - mean_v[:, :, None]
    numerator = (m * dt * dv).sum(2)
    denominator = (m * dt.square()).sum(2)
    return torch.where(count >= 2, numerator / denominator.clamp_min(1e-12), 0)

@torch.no_grad()
def frozen_body_inputs(model, data, device, batch):
    result = {k: [] for k in ['own', 'dynamics', 'initial_normal', 'event_probability', 'positive_atom', 'eligible']}
    for start in range(0, len(data['oi']), batch):
        sl = slice(start, start + batch)
        args = [torch.as_tensor(data[k][sl], device=device) for k in ['seq', 'anchor', 'support', 'installed']]
        out, info = joint_forward(model, args, True)
        own = model.encode(*args)[0][:, 1:]
        v, m, _, eligible = model.observation_streams(*args)
        v = v[:, 1:]
        m = m[:, 1:]
        slope8 = masked_slope(v, m, 8)
        slope4 = masked_slope(v, m, 4)
        pair = m[:, :, -7:] & m[:, :, -8:-1]
        amplitude = torch.where(pair, (v[:, :, -7:] - v[:, :, -8:-1]).abs(), 0).sum(2) / pair.sum(2).clamp_min(1)
        index = torch.where(m, torch.arange(24, device=device)[None, None, :, None], -1).max(2).values
        current = v.gather(2, index.clamp_min(0)[:, :, None]).squeeze(2) - 1
        anchor = args[1][..., PHYSICAL] - 1
        current = torch.where(index >= 0, current, torch.nan_to_num(anchor))
        age = torch.where(index >= 0, 23 - index, 24).to(v.dtype) / 24
        dynamics = torch.stack((current, slope8, amplitude, (slope4 - slope8) / 4, age, m[:, :, -8:].to(v.dtype).mean(2)), -1)
        state = info['state'][:, 1:]
        known = info['known'][:, 1:] & (state < 8)
        inferred_initial = info['raw_logits'][:, 1:, :, :8].softmax(-1)
        onehot = torch.nn.functional.one_hot(state.clamp(0, 7), 8).to(v.dtype)
        initial = torch.where(known[..., None], onehot, inferred_initial)
        packet = dict(own=own, dynamics=dynamics, initial_normal=initial, event_probability=out['probability'][:, 1:, ..., 8], positive_atom=out['atoms'][:, 1:, ..., 8], eligible=eligible[:, 1:])
        for key, value in packet.items():
            result[key].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}

def normalization_from_fit(packet):
    features = np.concatenate((packet['own'], packet['dynamics']), -1).astype('float64')
    eligible = packet['eligible'][..., None]
    count = eligible.sum((0, 1))
    if not (count > 0).all():
        raise ValueError('FIT support required for each physical parameter')
    mean = np.where(eligible, features, 0).sum((0, 1)) / count
    variance = np.where(eligible, (features - mean) ** 2, 0).sum((0, 1)) / count
    scale = np.maximum(np.sqrt(variance), 1e-05)
    return (mean.astype('float32'), scale.astype('float32'))
