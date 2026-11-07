"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from torch import nn
from torch.nn import functional as F
PHYSICAL = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
PRIMARY = [0, 1, 8, 9]

def asymmetric_transition(normal_logits, exit_logits, up_logits, down_logits):
    """(...,K,K), (...,K), (...,K), (...,1) -> (...,K+1,K+1)."""
    normal = normal_logits.softmax(-1)
    exit_probability = exit_logits.softmax(-1)
    up, down = (up_logits.sigmoid(), down_logits.sigmoid())
    top = torch.cat(((1 - up)[..., None] * normal, up[..., None]), -1)
    bottom = torch.cat((down * exit_probability, 1 - down), -1).unsqueeze(-2)
    return torch.cat((top, bottom), -2)

def observed_state(gap, edges, known):
    """Zero belongs to NORMAL because the original event definition is > U.

    Supports broadcasting edges to gap.shape+(K+1,). Unknowns remain -1.
    Values below -1 are invalid physical targets, not silently clipped.
    """
    if known.dtype != torch.bool or known.shape != gap.shape:
        raise ValueError('Explicit boolean known mask required')
    if (known & (~torch.isfinite(gap) | (gap < -1))).any():
        raise ValueError('Known state must be finite and physically nonnegative')
    safe = torch.where(known, gap, torch.zeros_like(gap))
    state = (safe[..., None] > edges[..., 1:-1]).sum(-1)
    state = torch.where(safe > 0, torch.full_like(state, edges.shape[-1] - 1), state)
    return torch.where(known, state, torch.full_like(state, -1))

class AsymmetricBoundaryFlux(nn.Module):
    """Common interface seq(B,6,24,28), anchor(B,6,11), support(B,6,11).

    Inputs contain only observed histories. Internal streams: workface then six
    regions. Every parameter retains its own encoder path. A workface history
    bin is known only when all currently installed regions for that parameter
    were observed in that bin. No uninstalled/missing placeholders enter values.

    Initial mass describes the COMMON last historical bin (issue_index-2),
    inferred from available history. Nine transitions produce targets at elapsed
    2..9, corresponding to original future leads 1..8, not elapsed 1..8.
    """

    def __init__(self, normal_edges, *, width):
        super().__init__()
        edges = torch.as_tensor(normal_edges, dtype=torch.float32).clone()
        if edges.shape != (10, 9) or not torch.all(edges[:, 0] == -1) or (not torch.all(edges[:, -1] == 0)) or (not torch.all(torch.diff(edges, dim=-1) > 0)):
            raise ValueError('Ten ordered normal-state geometries [-1,...,0] required')
        self.width = width
        self.register_buffer('normal_edges', edges)
        self.register_buffer('physical', torch.tensor(PHYSICAL))
        self.register_buffer('target_elapsed', torch.arange(2, 10))
        self.identity = nn.Embedding(10, width)
        self.temporal = nn.Sequential(nn.Conv1d(4, 16, 3, padding=1), nn.GELU(), nn.Conv1d(16, 16, 3, padding=2, dilation=2), nn.GELU())
        self.own_read = nn.Sequential(nn.Linear(16 * 24 + 3, width), nn.GELU())
        self.background = nn.Sequential(nn.Linear(24 * 8, width), nn.GELU())
        self.context_read = nn.Sequential(nn.Linear(width * 3, width), nn.GELU())
        self.clock = nn.Sequential(nn.Linear(2, width), nn.GELU())
        self.initial_mass = nn.Linear(width, 9)
        self.own_crossing = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))
        self.context_crossing = nn.Linear(width, 9)
        self.gate = nn.Linear(width * 2, 2)
        self.atoms = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))
        self.normal_destination = nn.Parameter(torch.randn(10, 8, 8) * 0.03)
        self.shared_exit_destination = nn.Parameter(torch.randn(8) * 0.03)

    def observation_streams(self, seq, anchor, support, installed):
        if seq.ndim != 4 or seq.shape[1:] != (6, 24, 28):
            raise ValueError('Expected six-region 24-bin input')
        if installed.shape != support.shape or support.shape != anchor.shape or anchor.shape[1:] != (6, 11):
            raise ValueError('Sensor context shapes disagree')
        mask = (seq[..., 11:22] > 0.5) & installed[:, :, None, :] & support[:, :, None, :]
        mask = mask & torch.isfinite(seq[..., :11])
        restored = seq[..., :11].clone()
        restored[..., 3:9] = restored[..., 3:9] + torch.nan_to_num(anchor[:, :, None, 3:9])
        restored = torch.where(mask, restored, torch.zeros_like(restored))
        own_region = restored[..., self.physical]
        own_mask = mask[..., self.physical]
        ins = installed[..., self.physical]
        face_mask = (own_mask | ~ins[:, :, None, :]).all(1) & ins.any(1)[:, None, :]
        face_value = torch.where(own_mask, own_region, -torch.inf).max(1).values
        face_value = torch.where(face_mask, face_value, torch.zeros_like(face_value))
        values = torch.cat((face_value[:, None], own_region), 1)
        observed = torch.cat((face_mask[:, None], own_mask), 1)
        face_support = (support[..., self.physical] | ~ins).all(1) & ins.any(1)
        eligibility = torch.cat((face_support[:, None], support[..., self.physical]), 1)
        controls = seq[..., 22:]
        if not torch.isfinite(controls).all():
            raise ValueError('Explicit control missingness policy required')
        micro = restored[..., 2]
        micro_mask = mask[..., 2]
        regional_background = torch.cat((controls, torch.asinh(micro)[..., None], micro_mask[..., None].to(seq.dtype)), -1)
        face_background = regional_background.mean(1, keepdim=True)
        background = torch.cat((face_background, regional_background), 1)
        return (values, observed, background, eligibility)

    def encode(self, seq, anchor, support, installed):
        values, observed, background, eligibility = self.observation_streams(seq, anchor, support, installed)
        b, s, t, c = values.shape
        gap = torch.where(observed, values - 1, torch.zeros_like(values))
        valid_difference = observed[:, :, 1:] & observed[:, :, :-1]
        change = torch.cat((torch.zeros_like(gap[:, :, :1]), torch.where(valid_difference, gap[:, :, 1:] - gap[:, :, :-1], 0)), 2)
        features = torch.stack((torch.asinh(gap), observed.to(seq.dtype), torch.asinh(change), torch.zeros_like(gap)), -1)
        clock = torch.linspace(-1, 0, t, device=seq.device, dtype=seq.dtype)
        features[..., 3] = torch.where(observed, clock[None, None, :, None], 0)
        packed = features.permute(0, 1, 3, 4, 2).reshape(b * s * c, 4, t)
        last = torch.where(observed, torch.arange(t, device=seq.device)[None, None, :, None], -1).max(2).values
        age = torch.where(last >= 0, 23 - last, 24).to(seq.dtype)
        final = gap.gather(2, last.clamp_min(0)[:, :, None]).squeeze(2)
        final = torch.where(last >= 0, final, torch.zeros_like(final))
        extra = torch.stack((torch.asinh(final), age / 24, observed.to(seq.dtype).mean(2)), -1)
        own = self.own_read(torch.cat((self.temporal(packed).flatten(1), extra.reshape(-1, 3)), -1)).reshape(b, s, c, self.width)
        own = own + self.identity.weight[None, None]
        eligible = eligibility[..., None].to(seq.dtype)
        total = (own * eligible).sum(2, keepdim=True)
        denominator = eligible.sum(2, keepdim=True) - eligible
        foreign = (total - own * eligible) / denominator.clamp_min(1)
        foreign = torch.where(denominator > 0, foreign, torch.zeros_like(foreign))
        background = self.background(torch.asinh(background).flatten(-2))[:, :, None].expand(-1, -1, c, -1)
        context = self.context_read(torch.cat((own, foreign, background), -1))
        return (own, context, eligibility, age)

    def forward(self, seq, anchor, support, installed, context_enabled=True, return_transitions=False, direction_enabled=None, readout='transport'):
        if readout not in ['transport', 'direct_distribution']:
            raise ValueError('Unknown probability readout')
        own, context, eligibility, age = self.encode(seq, anchor, support, installed)
        b, s, c, d = own.shape
        elapsed = torch.arange(1, 10, device=seq.device, dtype=seq.dtype)
        time_features = torch.stack((elapsed / 9, (elapsed / 9).square()), -1)
        own_step = own[..., None, :] + self.clock(time_features)[None, None, None]
        context_step = context[..., None, :].expand(-1, -1, -1, 9, -1)
        gates = self.gate(torch.cat((own_step, context_step), -1)).sigmoid()
        if not context_enabled:
            gates = torch.zeros_like(gates)
        if direction_enabled is not None:
            enabled = torch.as_tensor(direction_enabled, device=seq.device, dtype=seq.dtype)
            if enabled.shape != (2,) or not ((enabled == 0) | (enabled == 1)).all():
                raise ValueError('Two binary entry/exit route switches required')
            gates = gates * enabled
        correction = self.context_crossing(context_step)
        correction = torch.cat((correction[..., :8] * gates[..., :1], correction[..., 8:] * gates[..., 1:]), -1)
        crossing = self.own_crossing(own_step) + correction
        initial_logits = self.initial_mass(own)
        initial = initial_logits.softmax(-1)
        if readout == 'transport' or return_transitions:
            normal = self.normal_destination[None, None, :, None].expand(b, s, -1, 9, -1, -1)
            exit_logits = self.shared_exit_destination[None, None, None, None].expand(b, s, c, 9, -1)
            transition = asymmetric_transition(normal, exit_logits, crossing[..., :8], crossing[..., 8:])
        if readout == 'transport':
            mass = initial
            all_mass, entries, exits = ([], [], [])
            for step in range(9):
                u = crossing[..., step, :8].sigmoid()
                down = crossing[..., step, 8].sigmoid()
                entries.append((mass[..., :8] * u).sum(-1))
                exits.append(mass[..., 8] * down)
                mass = torch.matmul(mass.unsqueeze(-2), transition[..., step, :, :]).squeeze(-2)
                all_mass.append(mass)
            full_mass = torch.stack(all_mass, -2)
        else:
            normal_mixed = torch.einsum('bschi,cij->bschj', crossing[..., :8], self.normal_destination) / 8 ** 0.5
            positive_mixed = (crossing[..., :8] * self.shared_exit_destination).sum(-1, keepdim=True) / 8 ** 0.5
            logits = crossing + initial_logits[..., None, :] + torch.cat((normal_mixed, positive_mixed), -1)
            full_mass = logits.softmax(-1)
        probabilities = full_mass[..., 1:, :]
        atom_logits = self.atoms(own_step[..., 1:, :])
        edges = self.normal_edges[None, None, :, None]
        normal_atoms = edges[..., :-1] + torch.diff(edges, dim=-1) * atom_logits[..., :8].sigmoid()
        positive_atom = F.softplus(atom_logits[..., 8:]) + 1e-06
        atoms = torch.cat((normal_atoms, positive_atom), -1)
        mean = 1 + (probabilities * atoms).sum(-1)
        median_index = (probabilities.cumsum(-1) < 0.5).sum(-1).clamp_max(8)
        median = 1 + atoms.gather(-1, median_index[..., None]).squeeze(-1)
        result = dict(face_value=mean[:, 0].transpose(1, 2), regional_value=mean[:, 1:].transpose(-1, -2), face_median=median[:, 0].transpose(1, 2), face_risk=probabilities[:, 0, ..., 8].transpose(1, 2), regional_risk=probabilities[:, 1:, ..., 8].transpose(-1, -2), probability=probabilities, atoms=atoms, full_elapsed_probability=full_mass, initial_probability=initial, eligibility=eligibility, history_age=age, gates=gates)
        if readout == 'transport':
            result['entered'] = torch.stack(entries, -1)
            result['exited'] = torch.stack(exits, -1)
        if return_transitions:
            result['transition'] = transition
        return result
