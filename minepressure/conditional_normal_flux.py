"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from torch import nn
from torch.nn import functional as F

def directional_destinations(size, local, device=None, dtype=None):
    left = torch.zeros(size, size, device=device, dtype=dtype)
    right = torch.zeros_like(left)
    for i in range(size):
        if i:
            if local:
                left[i, i - 1] = 1
            else:
                left[i, :i] = 1 / i
        if i < size - 1:
            if local:
                right[i, i + 1] = 1
            else:
                right[i, i + 1:] = 1 / (size - i - 1)
    return (left, right)

def stochastic_directional_transition(left_rate, right_rate, local=True):
    """Boundary rates have already been explicitly zeroed by the caller."""
    if left_rate.shape != right_rate.shape or left_rate.shape[-1] != 8:
        raise ValueError('Eight normal-state left/right rates required')
    if (left_rate < 0).any() or (right_rate < 0).any():
        raise ValueError('Nonnegative rates required')
    if (left_rate[..., 0] != 0).any() or (right_rate[..., -1] != 0).any():
        raise ValueError('No flux outside normal-body state boundaries')
    denominator = 1 + left_rate + right_rate
    left = left_rate / denominator
    right = right_rate / denominator
    stay = 1 / denominator
    L, R = directional_destinations(8, local, left_rate.device, left_rate.dtype)
    return stay[..., None] * torch.eye(8, device=left_rate.device, dtype=left_rate.dtype) + left[..., None] * L + right[..., None] * R

class ConditionalNormalFlux(nn.Module):
    """B,region,10parameter interfaces; original leads1..8 are elapsed2..9.

    dynamics=[current_or_last_causal_gap,slope8,mean_abs_adjacent_change8,
              (slope4-slope8)/4,age/24,observed_fraction8].
    own is a frozen32-coordinate per-parameter encoding. Input normalization
    must be fitted only on eligible FIT histories. No targets are accepted.
    """

    def __init__(self, normal_edges, feature_mean, feature_scale, *, width, local):
        super().__init__()
        edges = torch.as_tensor(normal_edges, dtype=torch.float32)
        if edges.shape != (10, 9) or not (torch.diff(edges, dim=-1) > 0).all():
            raise ValueError('Ten ordered8state geometries')
        self.local = bool(local)
        self.register_buffer('normal_edges', edges.clone())
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        if mean.shape != (10, 38) or scale.shape != (10, 38) or (not (scale > 0).all()):
            raise ValueError('FIT-only feature geometry required')
        self.register_buffer('feature_mean', mean.clone())
        self.register_buffer('feature_scale', scale.clone())
        self.operator = nn.Sequential(nn.Linear(40, width), nn.GELU(), nn.Linear(width, 22))
        self.register_buffer('target_elapsed', torch.arange(2, 10))

    def forward(self, own, dynamics, initial_normal, event_probability, positive_atom):
        if own.ndim != 4 or own.shape[2:] != (10, 32) or dynamics.shape != (*own.shape[:-1], 6):
            raise ValueError('Expected own(B,R,10,32),dynamics(B,R,10,6)')
        if initial_normal.shape != (*own.shape[:-1], 8):
            raise ValueError('Explicit initial normal distribution required')
        if event_probability.shape != (*own.shape[:-1], 8) or positive_atom.shape != event_probability.shape:
            raise ValueError('Fixed per-parameter/per-lead tail probability and magnitude required')
        if ((event_probability < 0) | (event_probability > 1)).any() or not (positive_atom > 0).all():
            raise ValueError('Invalid fixed tail')
        if (initial_normal < 0).any() or not torch.allclose(initial_normal.sum(-1), torch.ones_like(initial_normal[..., 0]), atol=2e-06, rtol=0):
            raise ValueError('Initial normal conditional weights must be normalized')
        raw = torch.cat((own, dynamics), -1)
        normalized = ((raw - self.feature_mean[None, None]) / self.feature_scale[None, None]).clamp(-12, 12)
        elapsed = torch.arange(1, 10, device=own.device, dtype=own.dtype)
        clock = torch.stack((elapsed / 9, (elapsed / 9).square()), -1)
        expanded = normalized[..., None, :].expand(*normalized.shape[:-1], 9, 38)
        clock = clock[None, None, None].expand(*expanded.shape[:-1], 2)
        logits = self.operator(torch.cat((expanded, clock), -1))
        left_gate, right_gate, emission = (logits[..., :7], logits[..., 7:14], logits[..., 14:])
        amplitude = dynamics[..., 2].clamp_min(0)[..., None, None]
        velocity = dynamics[..., 1, None, None]
        centers = (self.normal_edges[:, :-1] + self.normal_edges[:, 1:]) * 0.5
        distance = torch.diff(centers, dim=-1)[None, None, :, None]
        left = (F.relu(-velocity) + amplitude * F.softplus(left_gate)) / distance
        right = (F.relu(velocity) + amplitude * F.softplus(right_gate)) / distance
        left = torch.cat((torch.zeros_like(left[..., :1]), left), -1)
        right = torch.cat((right, torch.zeros_like(right[..., :1])), -1)
        transition = stochastic_directional_transition(left, right, self.local)
        mass = initial_normal
        masses = []
        for step in range(9):
            mass = torch.matmul(mass.unsqueeze(-2), transition[..., step, :, :]).squeeze(-2)
            masses.append(mass)
        normal = torch.stack(masses, -2)[..., 1:, :]
        lower = self.normal_edges[None, None, :, None, :-1]
        upper = self.normal_edges[None, None, :, None, 1:]
        anchor = dynamics[..., 0, None, None]
        anchor = anchor.clamp(min=lower, max=upper)
        displacement = amplitude * elapsed[None, None, None, :, None] * emission.tanh()
        atoms = (anchor + displacement)[..., 1:, :].clamp(min=lower, max=upper)
        probability = torch.cat(((1 - event_probability[..., None]) * normal, event_probability[..., None]), -1)
        full_atoms = torch.cat((atoms, positive_atom[..., None]), -1)
        return dict(probability=probability, atoms=full_atoms, mean=1 + (probability * full_atoms).sum(-1), normal_probability=normal, normal_atoms=atoms, transition=transition, event_probability=event_probability)
