"""Paper algorithm components; no experiment execution or stored observations."""
import torch

def robust_square(error, delta):
    return 2 * error.square() / (torch.sqrt(1 + (error / delta).square()) + 1)
