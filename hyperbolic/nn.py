"""
    hyperbolic/nn.py
    Feb 22 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import numpy as np
import hyperbolic.functional as hf

import torch.nn.init as init


class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, dim, n_classes, k):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, dim))
        self.k = k
        self.n_classes = n_classes
        self.dim = dim
        self.reset_parameters()

    def forward(self, x, k=None):
        if k is None:
            k = torch.as_tensor(self.k).type_as(x)
        else:
            k = torch.as_tensor(k).type_as(x)

        p_vals_poincare = hf.poincare_exp0(self.p_vals, k)

        conformal_factor = 1 + k * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor

        logits = hf.hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, k)

        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.dim, self.n_classes, self.k)

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=np.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=np.sqrt(5))




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
    def __init__(self, k: float, riemannian: bool, clip: float=None):
        super(PoincareExp0, self).__init__()
       
        self.k = k
        self.riemannian = PoincareGradient
        self.riemannian.k = k
        self.clip = clip

        if riemannian:
            self.grad_correction = lambda x: self.riemannian.apply(x)
        else:
            self.grad_correction = lambda x: x

    def forward(self, u):
        x = hf.poincare_exp0(u, k=self.k, clip=self.clip)
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