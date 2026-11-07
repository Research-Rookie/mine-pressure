"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np, torch
from .invariant_phase_network import history_views
IDS = [0, 1, 9, 10]

@torch.no_grad()
def four_features(d, device, normalized_lower_limits, batch_size):
    rows = []
    for start in range(0, len(d['oi']), batch_size):
        seq = torch.as_tensor(d['seq'][start:start + batch_size], device=device)
        anchor = torch.as_tensor(d['anchor'][start:start + batch_size], device=device)
        basic, rich, _, _, age, _, _ = history_views(seq, anchor, normalized_lower_limits)
        f = basic[:, :, IDS]
        local = rich[:, :, IDS]
        aa = age[:, :, IDS].to(f.dtype)
        lead = []
        for step in range(2, 10):
            gap = aa + step
            level = f[..., 0]
            time = torch.stack([torch.full_like(gap, step / 9), *[level + gap * f[..., k] - 1 for k in [3, 4, 5]], level + gap * f[..., 4] + 0.5 * gap.square() * f[..., 6] - 1, gap / 32], -1)
            lead.append(torch.cat([local, time], -1))
        rows.append(torch.stack(lead, 2).cpu().numpy())
    return np.concatenate(rows)
