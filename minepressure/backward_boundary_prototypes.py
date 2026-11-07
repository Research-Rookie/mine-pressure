"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .kinematic_condition_network import FEATURE_IDS

def precursor_coordinates(theta, elapsed):
    """Map (...,K,6) canonical coefficients and broadcastable elapsed to 6 fields.

    theta=(endpoint_gap,v4,v8,acceleration,cubic_coefficient,difference4).
    v4/v8 are feature coefficients, not identified physical velocities.
    Returned fields have the original six-coordinate order. Elapsed includes
    actual observation age and the issue-to-target gap (2..9 for fresh input).
    """
    b, v4, v8, a, j, d = theta.unbind(-1)
    h = elapsed
    slope4 = v4 - h * a + 0.5 * h.square() * j
    slope8 = v8 - h * a + 0.5 * h.square() * j
    difference = d - 1.5 * h * a + 0.75 * h.square() * j
    margin8 = b - 0.5 * h.square() * a + h.pow(3) * j / 3
    margin4 = margin8 + h * (v4 - v8)
    curved = b - h.pow(3) * j / 6
    return torch.stack((slope4, slope8, difference, margin4, margin8, curved), -1)

class BackwardBoundaryConditions(nn.Module):

    def __init__(self, state, transport=True):
        super().__init__()
        self.transport = bool(transport)
        for key in ['mean', 'scale', 'center', 'width', 'canonical_center', 'canonical_scale']:
            self.register_buffer('initial_' + key, torch.as_tensor(state[key]).float().clone())
        self.center_delta = nn.Parameter(torch.zeros_like(self.initial_center))
        self.log_width = nn.Parameter(torch.zeros_like(self.initial_width))
        self.relevance = nn.Parameter(torch.zeros(6))
        self.rules = len(self.initial_center)
        self.offset = nn.Parameter(torch.full((self.rules,), 8.0))
        self.raw_temperature = nn.Parameter(torch.tensor(-3.0))
        self.bias = nn.Parameter(torch.tensor(-6.0))

    def conditions(self, x):
        if x.shape[-1] != 31 or not torch.isfinite(x).all():
            raise ValueError('Finite history descriptors required')
        z = (x[..., FEATURE_IDS] - self.initial_mean) / self.initial_scale
        if self.transport:
            theta = self.initial_canonical_center + 0.5 * torch.tanh(self.center_delta) * self.initial_canonical_scale
            h = x[..., 30, None] * 32
            if not (h > 0).all():
                raise ValueError('Elapsed clock must include positive forecast interval')
            center = (precursor_coordinates(theta, h) - self.initial_mean) / self.initial_scale
        else:
            center = self.initial_center + 0.5 * torch.tanh(self.center_delta)
        width = self.initial_width * torch.exp(self.log_width.clamp(-2, 2))
        temperature = 0.02 + F.softplus(self.raw_temperature)
        log_membership = F.logsigmoid((z[..., None, :] - (center - width)) / temperature) + F.logsigmoid((center + width - z[..., None, :]) / temperature)
        return (log_membership * F.softplus(self.relevance)).sum(-1)

    def forward(self, x):
        return self.bias + torch.logsumexp(self.conditions(x) + self.offset, -1) - np.log(self.rules)

class TwoBackwardBoundaryCovers(nn.Module):

    def __init__(self, state, transport=True):
        super().__init__()
        self.positive = BackwardBoundaryConditions(state, transport)
        self.second = BackwardBoundaryConditions(state, transport)
        with torch.no_grad():
            self.second.bias.fill_(-12.0)

    def forward(self, x):
        return self.positive(x) + F.softplus(self.second(x))
