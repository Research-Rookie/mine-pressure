"""Paper algorithm components; no experiment execution or stored observations."""
import numpy as np
import torch
from torch import nn

def numpy_projection(probability, gaps, requested_mean):
    p = np.asarray(probability, dtype=np.float64)
    a = np.asarray(gaps, dtype=np.float64)
    mu = np.asarray(requested_mean, dtype=np.float64)
    if p.shape != a.shape or mu.shape != p.shape[:-1]:
        raise ValueError('Incompatible distribution/mean shapes')
    if not (np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(mu).all()):
        raise ValueError('Finite inputs required')
    if p.min() < 0 or np.any(a < -1 - 1e-12) or np.any(mu < 0):
        raise ValueError('Nonnegative distribution/mean required')
    np.testing.assert_allclose(p.sum(-1), 1, atol=5e-12, rtol=0)
    normal = a <= 0
    n = (p * normal).sum(-1)
    body = (p * np.where(normal, 1 + a, 0)).sum(-1)
    tail = (p * np.where(~normal, 1 + a, 0)).sum(-1)
    target = np.minimum(np.maximum(mu, tail), tail + n)
    safe = np.where(n > 0, n, 1.0)
    old_average = np.clip(body / safe, 0, 1)
    new_average = np.clip((target - tail) / safe, 0, 1)
    down = new_average <= old_average
    scale = np.where(down, new_average / np.where(old_average > 0, old_average, 1.0), (1 - new_average) / np.where(old_average < 1, 1 - old_average, 1.0))
    scale = np.clip(scale, 0, 1)
    bias = np.where(down, 0, 1 - scale)
    scale = np.where(n > 0, scale, 1)
    bias = np.where(n > 0, bias, 0)
    mapped = np.clip(scale[..., None] * (1 + a) + bias[..., None] - 1, -1, 0)
    new_gaps = np.where(normal, mapped, a)
    mean = (p * (1 + new_gaps)).sum(-1)
    return dict(probability=p, gaps=new_gaps, mean=mean, projected_target=target, requested_mean=mu, normal_scale=scale, normal_bias=bias, lower=tail, upper=tail + n)

class TailPreservingNormalProjection(nn.Module):

    def forward(self, probability, gaps, requested_mean):
        p = probability.double()
        a = gaps.double()
        mu = requested_mean.double()
        normal = a <= 0
        n = (p * normal).sum(-1)
        body = (p * torch.where(normal, 1 + a, 0)).sum(-1)
        tail = (p * torch.where(~normal, 1 + a, 0)).sum(-1)
        target = torch.minimum(torch.maximum(mu, tail), tail + n)
        safe = torch.where(n > 0, n, torch.ones_like(n))
        before = (body / safe).clamp(0, 1)
        after = ((target - tail) / safe).clamp(0, 1)
        down = after <= before
        down_den = torch.where(before > 0, before, torch.ones_like(before))
        up_den = torch.where(before < 1, 1 - before, torch.ones_like(before))
        scale = torch.where(down, after / down_den, (1 - after) / up_den).clamp(0, 1)
        bias = torch.where(down, torch.zeros_like(scale), 1 - scale)
        scale = torch.where(n > 0, scale, torch.ones_like(scale))
        bias = torch.where(n > 0, bias, torch.zeros_like(bias))
        mapped = (scale[..., None] * (1 + a) + bias[..., None] - 1).clamp(-1, 0)
        new_gaps = torch.where(normal, mapped, a)
        return dict(probability=p, gaps=new_gaps, mean=(p * (1 + new_gaps)).sum(-1), projected_target=target, requested_mean=mu, normal_scale=scale, normal_bias=bias, lower=tail, upper=tail + n)
