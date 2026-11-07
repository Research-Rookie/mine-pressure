"""Paper algorithm components; no experiment execution or stored observations."""
import math
import torch
from torch import nn
PHYSICAL = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

def history_arrays(sequence, anchor):
    mask = sequence[..., 11:22] > 0.5
    level = torch.nan_to_num(anchor)
    offset = level.clone()
    offset[..., 3:9] = 0
    wave = torch.where(mask, sequence[..., :11] - offset[..., None, :], 0.0)
    t = torch.arange(24, device=sequence.device)
    last = torch.where(mask, t[None, None, :, None], -1).max(-2).values
    age = torch.where(last >= 0, 23 - last, 0).to(sequence.dtype)
    return (wave, mask, level, age)

class ConditionalDisplacementOperator(nn.Module):

    def __init__(self, statistics, width=48, modes=4):
        super().__init__()
        self.modes = modes
        for key, value in statistics.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer('physical', torch.tensor(PHYSICAL, dtype=torch.long))
        self.register_buffer('steps', torch.arange(2.0, 10.0))
        self.history = nn.Sequential(nn.Linear(24 * 28, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.summary = nn.Sequential(nn.Linear(277, width), nn.GELU())
        self.context = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        self.identity = nn.Embedding(10, 8)
        self.local = nn.Sequential(nn.Linear(width + 24 + 24 + 2 + 8, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.read = nn.Sequential(nn.Linear(32 + 2, 32), nn.GELU(), nn.Linear(32, modes + 1))
        self.operator = nn.Parameter(torch.randn(10, modes, 8, 24) * 0.03)
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self, sequence, summary, anchor, return_parts=False):
        wave, mask, level, age = history_arrays(sequence, anchor)
        normalized = wave / self.wave_scale
        controls = (sequence[..., 22:] - self.control_mean) / self.control_sd
        pack = torch.cat([normalized, mask.to(sequence.dtype), controls], -1)
        c = self.context(torch.cat([self.history(pack.flatten(-2)), self.summary(summary)], -1))
        own = normalized[..., self.physical].transpose(-1, -2)
        obs = mask[..., self.physical].transpose(-1, -2).to(sequence.dtype)
        lev = ((level - self.level_mean) / self.level_sd)[..., self.physical]
        ages = age[..., self.physical]
        shape = (*sequence.shape[:2], 10)
        identity = self.identity.weight.expand(*sequence.shape[:2], -1, -1)
        local = self.local(torch.cat([c[..., None, :].expand(*shape, c.shape[-1]), own, obs, lev[..., None], ages[..., None] / 24, identity], -1))
        step = self.steps.expand(*shape, 8)
        time = torch.stack([step / 9, (step + ages[..., None]) / 33], -1)
        raw = self.read(torch.cat([local[..., None, :].expand(*shape, 8, 32), time], -1))
        gate = raw[..., :self.modes].softmax(-1)
        directions = torch.einsum('brjt,jmht->brjhm', own, self.operator) / math.sqrt(24)
        displacement = ((directions * gate).sum(-1) + raw[..., -1]) * self.output_scale[self.physical][None, None, :, None]
        value = level[..., self.physical, None] + displacement
        value = value.transpose(-1, -2)
        if return_parts:
            return (value, dict(gate=gate, displacement=displacement, age=ages))
        return value

def fitting_statistics(d):
    with torch.no_grad():
        seq = torch.as_tensor(d['seq'])
        anchor = torch.as_tensor(d['anchor'])
        wave, mask, level, _ = history_arrays(seq, anchor)
        support = torch.as_tensor(d['support'])
        target = torch.as_tensor(d['target']).transpose(1, 2)
        out = {}
        out['wave_scale'] = torch.stack([wave[..., j][mask[..., j]].std().clamp_min(0.001) for j in range(11)])
        out['level_mean'] = torch.stack([level[..., j][support[..., j]].mean() for j in range(11)])
        out['level_sd'] = torch.stack([level[..., j][support[..., j]].std().clamp_min(0.001) for j in range(11)])
        controls = seq[..., 22:]
        out['control_mean'] = controls.mean((0, 1, 2))
        out['control_sd'] = controls.std((0, 1, 2)).clamp_min(0.001)
        delta = target - anchor[:, :, None, :]
        finite = torch.isfinite(delta) & support[:, :, None, :]
        out['output_scale'] = torch.stack([delta[..., j][finite[..., j]].std().clamp_min(0.001) for j in range(11)])
        return {k: v.numpy() for k, v in out.items()}
