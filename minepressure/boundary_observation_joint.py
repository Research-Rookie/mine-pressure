"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .asymmetric_boundary_flux_network import observed_state

def joint_forward(model, args, hard_observation=False, return_transitions=False):
    values, observed, _, eligible = model.observation_streams(*args)
    known = observed[:, :, -1] & eligible
    gap = values[:, :, -1] - 1
    state = observed_state(gap, model.normal_edges[None, None].expand(*gap.shape, -1), known)
    info = dict(known=known, state=state)
    hard_logits = torch.nn.functional.one_hot(state.clamp_min(0), 9).to(values.dtype).log()

    def capture_and_condition(module, inputs, logits):
        info['raw_logits'] = logits
        return torch.where(known[..., None], hard_logits, logits) if hard_observation else logits
    handle = model.initial_mass.register_forward_hook(capture_and_condition)
    try:
        out = model(*args, readout='transport', return_transitions=return_transitions)
    finally:
        handle.remove()
    return (out, info)

def current_state_loss(info, face_den, region_den):
    known = info['known']
    labels = info['state']
    logits = info['raw_logits']
    safe = labels.clamp_min(0)
    cell = torch.nn.functional.cross_entropy(logits.reshape(-1, 9), safe.reshape(-1), reduction='none').reshape_as(labels)
    cell = torch.where(known, cell, torch.zeros_like(cell))
    return 0.5 * cell[:, 0].mean() / face_den + 0.5 * cell[:, 1:].mean() / region_den

def current_observation_denominators(data):
    import numpy as np
    ids = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    installed = data['installed'][..., ids]
    regional = (data['seq'][:, :, -1, 11:22][..., ids] > 0.5) & installed & data['support'][..., ids]
    face = (regional | ~installed).all(1) & installed.any(1)
    if not face.any() or not regional.any():
        raise ValueError('No source observed-cutoff states')
    return dict(current_face_den=float(np.mean(face)), current_region_den=float(np.mean(regional)))
