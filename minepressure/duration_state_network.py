"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .trajectory_network import masked_kinematics

def initial_phase(sequence, anchor, buckets, normalized_lower_limits):
    feature, support, observation_age = masked_kinematics(sequence, anchor)
    values = sequence[..., :11].transpose(-1, -2).clone()
    values[..., 3:9, :] += torch.nan_to_num(anchor[..., 3:9, None])
    lower = anchor.new_tensor(normalized_lower_limits)
    phase = torch.where(anchor > 1, 1, torch.where(anchor < lower, 2, 0))
    hist_phase = torch.where(values > 1, 1, torch.where(values < lower[None, None, :, None], 2, 0))
    mask = sequence[..., 11:22].transpose(-1, -2) > 0.5
    index = 23 - observation_age[..., None] - torch.arange(24, device=anchor.device)
    hm = torch.gather(mask, -1, index.clamp_min(0)) & (index >= 0)
    hp = torch.gather(hist_phase, -1, index.clamp_min(0))
    same = (hm & (hp == phase[..., None])).long().cumprod(-1)
    duration = (same.sum(-1) - 1).clamp(0, buckets - 1)
    return (feature, support, observation_age, phase, duration)
