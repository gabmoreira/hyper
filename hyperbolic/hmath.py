"""
    hyperbolic/hmath.py
    Feb 22 2023
    Gabriel Moreira
    
    (Borrowed from geoopt)
"""

import torch

class Atanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

    
class Asinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5

def atanh(x):
    return Atanh.apply(x)


def asinh(x):
    return Asinh.apply(x)


def acosh(x, clamp=1e-5):
    x = x.clamp(min=1.0+clamp)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()