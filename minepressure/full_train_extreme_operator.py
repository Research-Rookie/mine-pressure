"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from torch import nn
from .family_conditioned_operator import FamilyConditionedOperator, Decoder

class FullTrainWideFaceOperator(FamilyConditionedOperator):

    def __init__(self, statistics):
        super().__init__(statistics, 'shared_wide')
        self.decoders = nn.ModuleList([Decoder(96, self.modes)])

    def forward(self, sequence, summary, anchor, support, return_parts=False):
        q = super().forward(sequence, summary, anchor)
        if not return_parts:
            return q
        mask = support[..., self.physical]
        face = q.masked_fill(~mask[:, :, None], -float('inf')).max(1).values
        face = torch.where(mask.any(1)[:, None], face, 0.0)
        return (q, dict(face=face))
