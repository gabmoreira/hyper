"""
    hyperbolic/nn.py
    Feb 22 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import hyperbolic.functional as hf


class PoincareGradient(torch.autograd.Function):
    # Placeholder
    k = -1

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        scale = (1 + PoincareGradient.k * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale
    
    
class PoincareExp0(nn.Module):
    def __init__(self, k: float, riemannian: bool):
        super(PoincareExp0, self).__init__()
       
        self.k = k
        self.riemannian = PoincareGradient
        self.riemannian.k = k

        if riemannian:
            self.grad_correction = lambda x: self.riemannian.apply(x)
        else:
            self.grad_correction = lambda x: x

    def forward(self, u):
        x = hf.poincare_exp0(u, k=self.k)
        return self.grad_correction(hf.project2poincare(x, k=self.k))


class LorentzExp0(nn.Module):
    def __init__(self, k: float):
        super(LorentzExp0, self).__init__()
        self.k = k
        
    def forward(self, u):
        x = hf.lorentz_exp0(u, k=self.k)
        return x
    

class LorentzInclusion(nn.Module):
    def __init__(self, k: float):
        super(LorentzInclusion, self).__init__()
        self.k = k
        
    def forward(self, u):
        x = hf.lorentz_inclusion(u, k=self.k)
        return x
    
    
class SphericalProjection(nn.Module):
    def __init__(self, k: float):
        super(SphericalProjection, self).__init__()
        self.k = k
        
    def forward(self, x):
        x = hf.spherical_projection(x, k=self.k)
        return x