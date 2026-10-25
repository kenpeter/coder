import torch
import torch.nn.functional as F


def log_softmax(z, dim=-1):
    zmax = z.amax(dim=dim, keepdim=True)
    shifted = torch.where(z == zmax, torch.zeros_like(z), z - zmax)
    others = torch.where(z == zmax, torch.full_like(z, float("-inf")), shifted)
    m = -torch.logsumexp(others, dim=dim, keepdim=True)
    return shifted - F.softplus(-m)


def softmax(z, dim=-1):
    return torch.exp(log_softmax(z, dim=dim))


def cross_entropy(z, target):
    # z: (N, K), target: (N,) long
    zt = z.gather(-1, target.unsqueeze(-1))
    is_t = torch.zeros_like(z, dtype=torch.bool)
    is_t.scatter_(-1, target.unsqueeze(-1), True)
    others = torch.where(is_t, torch.full_like(z, float("-inf")), z - zt)
    m = -torch.logsumexp(others, dim=-1)
    return F.softplus(-m)
"""Fixed softmax/log-softmax/cross-entropy (Barron et al.)."""
