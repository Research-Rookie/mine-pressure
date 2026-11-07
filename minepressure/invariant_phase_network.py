"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .duration_state_network import initial_phase

def history_views(sequence, anchor, normalized_lower_limits):
    basic, support, age, phase, duration = initial_phase(sequence, anchor, 9, normalized_lower_limits)
    values = sequence[..., :11].transpose(-1, -2)
    observed = sequence[..., 11:22].transpose(-1, -2) > 0.5
    last_index = 23 - age
    latest = values.gather(-1, last_index[..., None]).squeeze(-1)

    def mean(window):
        m = observed[..., -window:]
        v = values[..., -window:]
        return (m * v).sum(-1) / m.sum(-1).clamp_min(1)
    mean4, mean8, mean24 = (mean(4), mean(8), mean(24))
    m8 = observed[..., -8:]
    std8 = (((values[..., -8:] - mean8[..., None]).square() * m8).sum(-1) / m8.sum(-1).clamp_min(1)).sqrt()
    maximum = values.masked_fill(~observed, -float('inf')).max(-1).values
    minimum = values.masked_fill(~observed, float('inf')).min(-1).values
    pair = observed[..., -7:] & observed[..., -8:-1]
    positive = (values[..., -7:] > values[..., -8:-1]) & pair
    positive_fraction = positive.sum(-1) / pair.sum(-1).clamp_min(1)
    idx = last_index[..., None] - torch.arange(8, device=values.device)
    vv = values.gather(-1, idx.clamp_min(0))
    mm = observed.gather(-1, idx.clamp_min(0)) & (idx >= 0)
    increasing = (vv[..., :-1] > vv[..., 1:]) & mm[..., :-1] & mm[..., 1:]
    run = increasing.long().cumprod(-1).sum(-1) / 7
    absolute = values.clone()
    absolute[..., 3:9, :] += torch.nan_to_num(anchor[..., 3:9, None])
    above = ((absolute[..., -8:] > 1) & m8).sum(-1) / m8.sum(-1).clamp_min(1)
    tt = torch.arange(24, device=values.device)
    peak_index = torch.where(observed & (values == maximum[..., None]), tt, -1).max(-1).values
    peak_age = (last_index - peak_index).clamp_min(0) / 23
    agreement = torch.sign(basic[..., 3]) * torch.sign(basic[..., 4])
    lower = anchor.new_tensor(normalized_lower_limits)
    lower_gap = torch.where(torch.isfinite(lower), torch.nan_to_num(anchor) - lower, 0)
    extra = torch.stack([latest - mean4, latest - mean8, latest - mean24, maximum - latest, latest - minimum, std8, positive_fraction, run, above, peak_age, agreement, lower_gap], -1)
    extra = torch.nan_to_num(extra, nan=0, posinf=0, neginf=0).masked_fill(~support[..., None], 0)
    local = torch.cat([basic, extra], -1)
    dynamic = torch.stack([latest, mean4, mean8, mean24, std8, basic[..., 4], observed.float().mean(-1)], -1).flatten(-2)
    controls = sequence[..., 22:28]
    context = torch.cat([dynamic, controls[:, :, -1], controls[:, :, -4:].mean(-2), controls.mean(-2)], -1)
    assert local.shape[-1] == 25 and context.shape[-1] == 95
    return (basic, local, context, support, age, phase, duration)
