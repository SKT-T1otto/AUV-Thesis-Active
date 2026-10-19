import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def soft_update_group(pairs, tau, return_diff=False):
    """Stage every target tensor before any target network is written."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or (not math.isfinite(tau)) or (not 0 <= tau <= 1):
        raise ValueError('target coefficient must be finite and in [0,1]')
    (staged, differences) = ([], [])
    with torch.no_grad():
        for (target, source) in pairs:
            (targets, sources) = (list(target.parameters()), list(source.parameters()))
            if len(targets) != len(sources):
                raise ValueError('target/source parameter count mismatch')
            avg_diff = 0.0
            for (dst, src) in zip(targets, sources):
                if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                    raise ValueError('target/source parameter contract mismatch')
                if not bool(torch.isfinite(dst).all()) or not bool(torch.isfinite(src).all()):
                    raise FloatingPointError('nonfinite target/source parameter')
                value = dst.detach().clone().lerp_(src.detach(), tau)
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('target interpolation produced nonfinite parameters')
                if return_diff:
                    diff = torch.norm(dst.data - src.data, p=2).item()
                    if not math.isfinite(diff):
                        raise FloatingPointError('nonfinite target update diagnostic')
                    avg_diff += diff
                staged.append((dst, value))
            differences.append(avg_diff / max(len(targets), 1) if return_diff else None)
        for (dst, value) in staged:
            dst.copy_(value)
    return differences

def hard_update(target, source):
    (targets, sources) = (list(target.parameters()), list(source.parameters()))
    if len(targets) != len(sources):
        raise ValueError('target/source parameter count mismatch')
    staged = []
    for (dst, src) in zip(targets, sources):
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise ValueError('target/source parameter contract mismatch')
        if not bool(torch.isfinite(src).all()):
            raise FloatingPointError('nonfinite hard-update source')
        staged.append(src.detach().clone())
    with torch.no_grad():
        for (dst, value) in zip(targets, staged):
            dst.copy_(value)

def onehot_from_logits(logits, eps=0.0):
    argmax_acs = torch.zeros_like(logits, dtype=torch.float32).scatter_(1, logits.argmax(1, keepdim=True), 1.0)
    if eps == 0.0:
        return argmax_acs
    random_indices = torch.as_tensor(np.random.choice(range(logits.shape[1]), size=logits.shape[0]), device=logits.device)
    rand_acs = Variable(torch.eye(logits.shape[1], device=logits.device)[random_indices], requires_grad=False)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for (i, r) in enumerate(torch.rand(logits.shape[0], device=logits.device))])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor, *, device=None, dtype=None):
    values = tens_type(*shape) if device is None and dtype is None else torch.empty(shape, device=device, dtype=dtype)
    U = Variable(values.uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
