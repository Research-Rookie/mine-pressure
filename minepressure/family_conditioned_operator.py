"""Paper algorithm components; no experiment execution or stored observations."""
import math
import torch
from torch import nn
from .conditional_displacement_operator import ConditionalDisplacementOperator, history_arrays
MAPS = {'family': [0, 0, 2, 2, 2, 2, 2, 2, 1, 1], 'mixed': [0, 1, 2, 2, 2, 2, 2, 2, 0, 1], 'shared_wide': [0] * 10}

class Decoder(nn.Module):

    def __init__(self, width, modes):
        super().__init__()
        self.local = nn.Sequential(nn.Linear(106, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.read = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, modes + 1))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

class FamilyConditionedOperator(ConditionalDisplacementOperator):

    def __init__(self, statistics, mode):
        super().__init__(statistics)
        assert mode in MAPS
        self.mode = mode
        del self.local
        del self.read
        self.register_buffer('family_index', torch.tensor(MAPS[mode], dtype=torch.long))
        self.decoders = nn.ModuleList([Decoder(68 if mode == 'shared_wide' else 32, self.modes) for _ in range(1 if mode == 'shared_wide' else 3)])

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
        features = torch.cat([c[..., None, :].expand(*shape, c.shape[-1]), own, obs, lev[..., None], ages[..., None] / 24, identity], -1)
        step = self.steps.expand(*shape, 8)
        time = torch.stack([step / 9, (step + ages[..., None]) / 33], -1)
        raw = features.new_zeros((*shape, 8, self.modes + 1))
        for k, decoder in enumerate(self.decoders):
            jj = torch.nonzero(self.family_index == k, as_tuple=True)[0]
            local = decoder.local(features[..., jj, :])
            h = local.shape[-1]
            raw[..., jj, :, :] = decoder.read(torch.cat([local[..., None, :].expand(*local.shape[:-1], 8, h), time[..., jj, :, :]], -1))
        gate = raw[..., :self.modes].softmax(-1)
        directions = torch.einsum('brjt,jmht->brjhm', own, self.operator) / math.sqrt(24)
        displacement = ((directions * gate).sum(-1) + raw[..., -1]) * self.output_scale[self.physical][None, None, :, None]
        value = (level[..., self.physical, None] + displacement).transpose(-1, -2)
        if return_parts:
            return (value, dict(gate=gate, displacement=displacement, age=ages))
        return value
