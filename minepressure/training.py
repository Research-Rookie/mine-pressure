"""Component objectives and optimizer operations; no experiment runner or logs."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .base_objective import loss as base_distribution_loss
from .body_objective import numeric_loss as body_distribution_loss
from .boundary_observation_joint import current_state_loss, joint_forward
from .numeric_robust_objective import robust_square

PRIMARY = [0, 1, 8, 9]
SECONDARY = [2, 3, 4, 5, 6, 7]


def parameter_mass(values, primary_mass):
    return primary_mass * values[PRIMARY].mean() + (1 - primary_mass) * values[SECONDARY].mean()


def prepare_numeric_targets(regional, regional_known, face, face_known, *, normalized_lower_limits, exceedance_weight):
    """Normalized full-TRAIN targets; preserve unknown masks and global denominators."""
    lower = regional.new_tensor(normalized_lower_limits)
    rw = regional_known*((regional > 1) | (regional < lower)).to(regional.dtype)*(exceedance_weight - 1)
    rw = rw+regional_known
    fw = face_known*((face > 1) | (face < lower)).to(face.dtype)*(exceedance_weight - 1)+face_known
    return dict(ry=torch.where(regional_known, regional, 0), rw=rw,
                fy=torch.where(face_known, face, 0), fw=fw), dict(
        rw_den=rw.mean((0, 1, 2)).clamp_min(1e-9),
        fw_den=fw.mean((0, 1)).clamp_min(1e-9))


def numerical_loss(regional_prediction, face_prediction, targets, statistics, delta, *, primary_mass):
    r = (robust_square(regional_prediction-targets['ry'], delta)*targets['rw']).mean((0, 1, 2))/statistics['rw_den']
    f = (robust_square(face_prediction-targets['fy'], delta)*targets['fw']).mean((0, 1))/statistics['fw_den']
    return .5*(parameter_mass(r, primary_mass)+parameter_mass(f, primary_mass))


def warning_loss(logits, positive, known, positive_weight, valid_denominator,
                 origin_importance):
    cell = F.binary_cross_entropy_with_logits(
        logits, positive.to(logits.dtype),
        pos_weight=logits.new_tensor(positive_weight), reduction='none')
    per_origin = torch.where(known, cell, 0).mean((1, 2))/valid_denominator
    return (per_origin*origin_importance).mean()


def positive_origin_proposal(has_positive, *, positive_mass):
    positive = np.asarray(has_positive, bool)
    assert positive.any()
    n = len(positive)
    proposal = np.full(n, (1 - positive_mass)/n)
    proposal[positive] += positive_mass/positive.sum()
    return proposal, 1/(n*proposal)


def warning_training_statistics(positive, known, *, weight_cap):
    positive_count = int((positive & known).sum())
    assert positive_count > 0
    weight = min(weight_cap, float((~positive & known).sum())/positive_count)
    return dict(positive_weight=weight,
                valid_denominator=float(known.float().mean()))


def make_optimizer(module, learning_rate, *, weight_decay):
    return torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)


def numerical_learning_rate(epoch, epochs, *, start_rate, end_rate):
    return end_rate+(start_rate-end_rate)*.5*(1+math.cos(math.pi*epoch/(epochs-1)))


def optimizer_step(module, optimizer, objective, *, max_gradient_norm):
    assert torch.isfinite(objective)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_gradient_norm, error_if_nonfinite=True)
    optimizer.step()
