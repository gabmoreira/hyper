"""
    hyperbolic/nn.py
    Feb 22 2023
    Gabriel Moreira
"""
import torch.nn as nn
import functional as hf

class PoincareExp0(nn.Module):
    def __init__(self, 
                 k: float,
                 dim: int,
                 riemannian: bool):
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
        return self.grad_fix(hf.project2poincare(x), k=self.k)

