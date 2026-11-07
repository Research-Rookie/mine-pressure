"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .asymmetric_boundary_flux_network import PRIMARY
from .boundary_flux_operator import atomic_crps, observed_transition_nll

def loss(out, t, stats, *, parameter_weights, numeric_weight, risk_weight, transition_weight):
    face_crps = atomic_crps(out['probability'][:, 0], out['atoms'][:, 0], t['fy'])
    region_crps = atomic_crps(out['probability'][:, 1:], out['atoms'][:, 1:], t['ry'])
    face = (face_crps * t['fm']).mean((0, 2)) / stats['face_den']
    region = (region_crps * t['rm']).mean((0, 1, 3)) / stats['region_den']
    weight = face.new_tensor(parameter_weights)
    numeric = ((face + region) * 0.5 * weight).sum()
    fp = out['probability'][:, 0, ..., 8]
    rp = out['probability'][:, 1:, ..., 8]
    fb = ((fp - (t['fy'] > 0).float()).square() * t['fm']).mean((0, 2)) / stats['face_den']
    rb = ((rp - (t['ry'] > 0).float()).square() * t['rm']).mean((0, 1, 3)) / stats['region_den']
    risk = ((fb + rb) * 0.5 * stats['risk_scale'])[PRIMARY].mean()
    pair = observed_transition_nll(out['transition'][:, 0], t['previous'], t['following'], t['pair_known'])
    pair = pair * t['pair_known'].float().mean() / stats['pair_den']
    return numeric_weight * numeric + risk_weight * risk + transition_weight * pair
