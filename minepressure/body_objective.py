"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .regional_maximum_distribution import maximum_atomic_distribution
from .boundary_flux_operator import atomic_crps

def numeric_loss(out, eligible, target, stats, *, parameter_weights):
    p = out['probability'].permute(0, 2, 3, 1, 4)
    a = out['atoms'].permute(0, 2, 3, 1, 4)
    active = eligible.transpose(1, 2)[:, :, None, :].expand(*p.shape[:-1])
    q = torch.ones(p.shape[:-2], device=p.device, dtype=torch.float64)
    face = maximum_atomic_distribution(p.double(), a.double(), active, q)
    fc = atomic_crps(face['probability'], face['atoms'], target['fy'])
    rc = atomic_crps(out['probability'], out['atoms'], target['ry'])
    f = (fc * target['fm']).mean((0, 2)) / stats['face_den']
    region = (rc * target['rm']).mean((0, 1, 3)) / stats['region_den']
    weight = f.new_tensor(parameter_weights)
    return (((f + region) * 0.5 * weight).sum(), face)
