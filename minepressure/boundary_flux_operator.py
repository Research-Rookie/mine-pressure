"""Paper algorithm components; no experiment execution or stored observations."""
import torch

def atomic_crps(probability, atoms, target):
    """Per-observation CRPS for an explicit atomic distribution.

    Values and targets must share units. Unknown targets must be removed by
    caller before reduction. A small CRPS does not prove risk calibration.
    """
    first = (probability * (atoms - target[..., None]).abs()).sum(-1)
    pair_distance = (atoms[..., :, None] - atoms[..., None, :]).abs()
    second = (probability[..., :, None] * probability[..., None, :] * pair_distance).sum((-1, -2))
    return first - 0.5 * second

def observed_transition_nll(transition, previous_state, next_state, known):
    """Loss ONLY on pairs with both observed adjacent physical states.

    Conditioning on previous_state here selects a likelihood row as a target;
    it must never enter an inference-time context encoder as future input.
    Unknown states may be -1 and are excluded, not clamped into a real class.
    """
    if known.shape != transition.shape[:-2]:
        raise ValueError('Pair mask must match transition leading dimensions')
    if known.dtype != torch.bool:
        raise ValueError('Pair mask must be boolean')
    if not known.any():
        return transition.sum() * 0
    t = transition[known]
    left, right = (previous_state[known], next_state[known])
    if ((left < 0) | (left >= t.shape[-1]) | (right < 0) | (right >= t.shape[-1])).any():
        raise ValueError('Known pair has invalid state index')
    prob = t[torch.arange(len(t), device=t.device), left, right]
    return -prob.clamp_min(torch.finfo(prob.dtype).tiny).log().mean()
