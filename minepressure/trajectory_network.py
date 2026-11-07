"""Paper algorithm components; no experiment execution or stored observations."""
import torch

def masked_kinematics(sequence, anchor):
    """[B,R,24,28], last finite physical anchor [B,R,J] in threshold units.

    Cumulative sequence channels are centered on the last finite value by the
    existing cache; regression slopes are translation invariant. Absolute levels
    always come from the uncentered causal persistence anchor.
    """
    values = sequence[..., :11].transpose(-1, -2)
    mask = sequence[..., 11:22].transpose(-1, -2) > 0.5
    t = torch.arange(-23, 1, device=sequence.device, dtype=sequence.dtype)
    last_index = torch.where(mask, t, t.new_full((), -24)).max(-1).values
    supported = mask.any(-1) & torch.isfinite(anchor)
    age = torch.where(supported, -last_index, 0).long()
    slopes, residuals, coverages = ([], [], [])
    for window in (4, 8, 24):
        y = values[..., -window:]
        m = mask[..., -window:].to(values.dtype)
        tt = t[-window:]
        n = m.sum(-1).clamp_min(1)
        tm = (m * tt).sum(-1) / n
        ym = (m * y).sum(-1) / n
        dt = tt - tm[..., None]
        denom = (m * dt.square()).sum(-1)
        slope = (m * dt * (y - ym[..., None])).sum(-1) / denom.clamp_min(1e-08)
        slope = torch.where(denom > 0, slope, torch.zeros_like(slope))
        residual = ((m * (y - ym[..., None] - slope[..., None] * dt).square()).sum(-1) / n).sqrt()
        slopes.append(slope)
        residuals.append(residual)
        coverages.append(n / window * mask[..., -window:].any(-1))
    safe_anchor = torch.nan_to_num(anchor)
    curvature = torch.where((coverages[0] * 4 >= 2) & (coverages[1] * 8 >= 2), (slopes[0] - slopes[1]) / 4, torch.zeros_like(slopes[0]))
    feature = torch.stack([safe_anchor, 1 - safe_anchor, age / 24, *slopes, curvature, *residuals, *coverages], -1)
    return (feature, supported, age)
