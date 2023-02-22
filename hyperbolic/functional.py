"""
    hyperbolic/functional.py
    Feb 22 2023
    Gabriel Moreira
"""
import numpy as np
import torch
import torch.nn.functional as F
from hmath import atanh, acosh


def euclidean_dist(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False):
    """
        Euclidean distance
    """
    dist = torch.norm(x - y, p=2, dim=-1, keepdim=keepdim)
    return dist


def poincare_dist(x: torch.Tensor, y: torch.Tensor, k: float, keepdim: bool = False):
    """
        Poincaré (curvature k < 0) distance
    """
    sqrt_k = (-k) ** 0.5
    dist = atanh(sqrt_c * mobius_add(-x, y, k).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, k: float, keepdim: bool = False):
    """
        Lorentz (curvature k < 0) distance
    """
    inner = lorentz_inner(x,y,keepdim)
    dist = (1.0 / np.sqrt(-k)) * acosh(k * inner)
    return dist


def euclidean_cdist(x: torch.Tensor, y: torch.Tensor):
    """
        Euclidean pairwise distance matrix
    """
    cdist = torch.cdist(x, y, p=2)
    return cdist


def poincare_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Poincaré pairwise distance matrix
    """
    sqrt_k = (-k) ** 0.5
    cdist =  (2.0/sqrt_k) * atanh(sqrt_k * torch.norm(mobius_addition_batch(-x, y, k), dim=-1))


def lorentz_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Lorentz pairwise distance matrix
    """ 
    metric = torch.ones((x.shape[-1]))
    metric[...,-1] = -1
    inner = torch.einsum('ik,jk,k->ij',x,y,metric)
    cdist = (1.0 / np.sqrt(-k)) * acosh(k*inner)
    return 


def lorentz_inner(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False):
    """
        Inner product defined from the Lorentz pseudometric
    """
    metric = torch.ones((x.shape[-1],)
    metric[-1] = -1
    inner = torch.einsum('...i,...i,i->...'x,y,metric)
    if keepdim:
        inner = inner.unsqueeze(-1)
    return inner

                        
def poincare_exp0(u : torch.Tensor, k: float):
    """
        Poincaré (curvature k < 0) exponential map @ 0
    """
    sqrt_k = (-k) ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma  = tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
    return gamma

                        
def sphere_projection(x: torch.Tensor, k: float):
    r = 1.0 / (-k)**2
    x = r * F.normalize(x, p=2, dim=-1)
    return x
                        
                        
def lorentz_inclusion(x: torch.Tensor, k: float):      
    """
        Hyperboloid (curvature k < 0) inclusion map
    """
    new_size     = list(x.shape)
    new_size[-1] = new_size[-1] + 1

    xz = torch.zeros(new_size)
    xz[...,:-1] += x
    xz[...,-1]  += torch.sqrt(torch.square(local_coordinates).sum(dim=-1) - 1.0/k)
    return xz

                        

def mobius_add(x: torch.Tensor, y: torch.Tensor, k: float):
    """
    """
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy - k ** 2 * x2 * y2
    return num / (denom + 1e-5)


def mobius_addition_batch(x: torch.Tensor, y: torch.Tensor, k: float):
    """
    """
    xy = torch.einsum("ij,kj->ik", x, y) # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 - 2 * k * xy - k * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 + k * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 - 2 * k * xy  # B x C
    denom_part2 = (-k) ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res
                        

def project2poincare(x: torch.Tensor, k: float, eps: float=1e-3):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1.0 - eps) / ((-k) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)
                        
                        
def mobius_transform(p, k):
    """
        Returns an hyperbolic isometry that sends p to the origin
    """
    assert(k < 0)
    p = p if isinstance(p, complex) else complex(*p)
    
    r = 1 / np.sqrt(-k)
    transform = lambda z : (z - p) / (1 - np.conj(p/r) * z/r)
    return transform


def mobius_inverse(p, k):
    """
        Inverse hyperbolic isometry of the Mobius transform
        Maps the origin back to p
    """
    assert(k < 0)
    p = p if isinstance(q, complex) else complex(*p)

    r = 1 / np.sqrt(-k)
    transform = lambda w : (w + p) / (1 + np.conj(p/r) * w/r)
    return transform


