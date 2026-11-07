"""Paper algorithm components; no experiment execution or stored observations."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def gaussian_bin_law(location, scale, edges):
    lower = edges[..., :-1]
    upper = edges[..., 1:]
    lo = (lower - location[..., None]) / scale[..., None]
    hi = (upper - location[..., None]) / scale[..., None]
    left = 0.5 * (torch.erfc(-hi / math.sqrt(2)) - torch.erfc(-lo / math.sqrt(2)))
    right = 0.5 * (torch.erfc(lo / math.sqrt(2)) - torch.erfc(hi / math.sqrt(2)))
    mass = torch.where(lo >= 0, right, left).clamp_min(0)
    density = (torch.exp(-0.5 * lo.square()) - torch.exp(-0.5 * hi.square())) / math.sqrt(2 * math.pi)
    mean = location[..., None] + scale[..., None] * density / mass.clamp_min(1e-12)
    atoms = torch.where(mass > 1e-12, mean, (lower + upper) / 2).clamp(min=lower, max=upper)
    probability = mass / mass.sum(-1, keepdim=True)
    return (probability, atoms)

class ConditionalGaussianBody(nn.Module):

    def __init__(self, normal_edges, feature_mean, feature_scale):
        super().__init__()
        self.register_buffer('normal_edges', torch.as_tensor(normal_edges, dtype=torch.float32).clone())
        self.register_buffer('feature_mean', torch.as_tensor(feature_mean, dtype=torch.float32).clone())
        self.register_buffer('feature_scale', torch.as_tensor(feature_scale, dtype=torch.float32).clone())
        self.operator = nn.Sequential(nn.Linear(48, 40), nn.GELU(), nn.Linear(40, 2))
        with torch.no_grad():
            self.operator[-1].bias[1] = -2.0

    def forward(self, own, dynamics, initial_normal, event_probability, positive_atom):
        normalized = ((torch.cat((own, dynamics), -1) - self.feature_mean) / self.feature_scale).clamp(-12, 12)
        initial = initial_normal[..., None, :].expand(*initial_normal.shape[:-1], 8, 8)
        history = normalized[..., None, :].expand(*normalized.shape[:-1], 8, 38)
        elapsed = torch.arange(2, 10, device=own.device, dtype=own.dtype)
        clock = torch.stack((elapsed / 9, (elapsed / 9).square()), -1)
        clock = clock.expand(*history.shape[:-1], 2)
        raw = self.operator(torch.cat((history, initial, clock), -1)).double()
        edges = self.normal_edges.double()[None, None, :, None, :]
        span = edges[..., -1] - edges[..., 0]
        centers = (self.normal_edges[:, :-1] + self.normal_edges[:, 1:]) * 0.5
        anchor = (initial_normal * centers).sum(-1).double()[..., None]
        fraction = ((anchor - edges[..., 0]) / span).clamp(0.001, 0.999)
        location = edges[..., 0] + span * torch.sigmoid(torch.logit(fraction) + raw[..., 0])
        scale = span * (0.02 + F.softplus(raw[..., 1]))
        normal, atoms = gaussian_bin_law(location, scale, edges)
        tail = event_probability.double()
        probability = torch.cat(((1 - tail[..., None]) * normal, tail[..., None]), -1)
        full_atoms = torch.cat((atoms, positive_atom.double()[..., None]), -1)
        return dict(probability=probability, atoms=full_atoms, mean=1 + (probability * full_atoms).sum(-1), normal_probability=normal, normal_atoms=atoms, event_probability=tail, location=location, scale=scale)
