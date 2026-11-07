"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from torch import nn
from torch.nn import functional as F
from .full_system_seed_repeat_helpers import numeric_initialization

class FaithfulPressureControl(nn.Module):

    def __init__(self, statistics, seed):
        super().__init__()
        self.mean = numeric_initialization(statistics, seed)
        torch.manual_seed(seed)
        self.scale = nn.Sequential(nn.Linear(98, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)
        self._features = None

        def capture(module, args):
            self._features = args[0]
        self.mean.decoders[0].read.register_forward_pre_hook(capture)

    def forward(self, seq, summary, anchor, support):
        value, parts = self.mean(seq, summary, anchor, support, True)
        features = self._features
        self._features = None
        active = support[..., self.mean.physical]
        pooled = features.detach().masked_fill(~active[..., None, None], -torch.inf).max(1).values
        pooled = torch.where(active.any(1)[..., None, None], pooled, 0.0)
        unit = self.mean.output_scale[self.mean.physical].clamp_min(0.001)
        sigma = F.softplus(self.scale(features.detach()).squeeze(-1)).transpose(-1, -2) * unit + 0.0001
        face_sigma = F.softplus(self.scale(pooled).squeeze(-1)).transpose(-1, -2) * unit + 0.0001
        return dict(regional_mean=value, face_mean=parts['face'], regional_sigma=sigma, face_sigma=face_sigma)

def losses(out, target, stats):
    primary = [0, 1, 8, 9]
    other = [2, 3, 4, 5, 6, 7]

    def mass(v):
        return 0.8 * v[primary].mean() + 0.2 * v[other].mean()
    regional = ((out['regional_mean'] - target['ry']).square() * target['rw']).mean((0, 1, 2)) / stats['rw_den']
    face = ((out['face_mean'] - target['fy']).square() * target['fw']).mean((0, 1)) / stats['fw_den']
    mean_loss = 0.5 * (mass(regional) + mass(face))

    def nll(y, mu, sigma):
        return 0.5 * ((y - mu.detach()) / sigma).square() + sigma.log() + 0.5 * __import__('math').log(2 * __import__('math').pi)
    regional = (nll(target['ry'], out['regional_mean'], out['regional_sigma']) * target['rm']).mean((0, 1, 2)) / stats['rm_den']
    face = (nll(target['fy'], out['face_mean'], out['face_sigma']) * target['fm']).mean((0, 1)) / stats['fm_den']
    return (mean_loss, 0.5 * (mass(regional) + mass(face)))
