"""
    hyperbolic/functional.py
    Feb 28 2023
    Gabriel Moreira
"""
import numpy as np
import torch
import torch.nn.functional as F

from .hmath import atanh, acosh, tanh, cosh, sinh, asinh



def hyperbolic_softmax(X, A, P, k):
    # Conformal factor
    lambda_pkc = 2 / (1 + k * P.pow(2).sum(dim=1))
    kk = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(abs(k))
    mob_add = mobius_addition_batch(-P, X, k)
    num = 2 * torch.sqrt(abs(k)) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = torch.norm(A, dim=1, keepdim=True) * (1 + k * mob_add.pow(2).sum(dim=2))
    logit = kk.unsqueeze(1) * asinh(num / denom)
    return logit.permute(1, 0)



"""
    -------------------------------------------------
    METRICS, DISTANCES AND PAIRWISE DISTANCE MATRICES
    -------------------------------------------------
"""

def lorentz_inner(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False):
    """
        Inner product defined from the Lorentz pseudometric
    """
    metric = torch.ones((x.shape[-1],), device=x.device)
    metric[-1] = -1
    inner = torch.einsum('ik,jk,k->ij',x,y,metric)
    if keepdim:
        inner = inner.unsqueeze(-1)
    return inner


def euclidean_dist(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False):
    """
        Euclidean distance
    """
    dist = torch.norm(x - y, p=2, dim=-1, keepdim=keepdim)
    return dist


def squared_euclidean_dist(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False):
    """
        Squared euclidean distance
    """
    dist = torch.norm(x - y, p=2, dim=-1, keepdim=keepdim)**2
    return dist


def poincare_dist(x: torch.Tensor, y: torch.Tensor, k: float, keepdim: bool = False):
    """
        Poincaré (curvature k < 0) distance
    """
    sqrt_k = (-k) ** 0.5
    dist = atanh(sqrt_k * mobius_add(-x, y, k).norm(dim=-1, p=2, keepdim=keepdim))
    return dist * 2 / sqrt_k


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, k: float, keepdim: bool = False):
    """
        Lorentz (curvature k < 0) distance
    """
    inner = lorentz_inner(x,y,keepdim)
    dist = (1.0 / np.sqrt(-k)) * acosh(k * inner)
    return dist


def euclidean_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Euclidean pairwise distance matrix
    """
    cdist = torch.cdist(x, y, p=2)
    return cdist


def squared_euclidean_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Squared Euclidean pairwise distance matrix
    """
    cdist = torch.cdist(x, y, p=2)**2
    return cdist


def poincare_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Poincaré pairwise distance matrix
    """
    sqrt_k = (-k) ** 0.5
    cdist  = atanh(sqrt_k * torch.norm(mobius_addition_batch(-x, y, k), dim=-1))
    cdist  = (2.0/sqrt_k) * cdist
    return cdist


def lorentz_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Lorentz pairwise distance matrix
    """ 
    m = torch.ones(x.shape[-1], device=x.device)
    m[-1] = -1
    inner_products = torch.einsum('ik,k,jk->ij', x, m, y)
    cdist = ((-1.0/k)**0.5) * acosh(k * inner_products)
    return cdist


def spherical_cdist(x: torch.Tensor, y: torch.Tensor, k: float):
    """
        Spherical pairwise distance matrix
    """ 
    r = 1/np.sqrt(k)
    inner = torch.clamp(F.normalize(x, dim=-1, p=2) @ F.normalize(x, dim=-1, p=2).permute(1,0), min=-1.0, max=1.0)
    cdist = r * torch.acos(inner)
    return cdist
    

def cdist(manifold: str, k: float):
    return eval('lambda x, y : ' + manifold + '_cdist(x,y,' + str(k) + ')')



"""
    ------------------------------------------
    EXPONENTIAL MAPS, INCLUSIONS & PROJECTIONS
    ------------------------------------------
"""
def lorentz2poincare(x : torch.Tensor, k: float):
    """
    """
    sk = np.sqrt(-k)
    y  = x[...,:-1] / (1.0 + sk * x[...,-1].unsqueeze(-1))
    y  = project2poincare(y, k)
    return y


def poincare2lorentz(x : torch.Tensor, k: float):
    """
    """
    sk  = np.sqrt(-k)
    lbd = 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)
    z   = 1.0 / sk * (lbd - 1.0)
    y   = torch.cat((lbd * x, z), dim=-1)
    return y


def poincare_exp0(u : torch.Tensor, k : float, clip: float=None):
    """
        Poincaré (curvature k < 0) exponential map @ 0
    """
    sqrt_k = (-k) ** 0.5

    if clip is not None:
        u = torch.min(torch.ones_like(u), clip / u.norm(dim=-1, p=2, keepdim=True)) * u

    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma  = tanh(sqrt_k * u_norm) * u / (sqrt_k * u_norm)
    return gamma


def lorentz_exp0(u : torch.Tensor, k : float):
    """
        Lorentz (curvature k < 0) exponential map @ 0
    """
    norm_u = torch.norm(u, p=2, dim=-1, keepdim=True)
    csh = cosh(norm_u * ((-k)**0.5))
    snh = sinh(norm_u * ((-k)**0.5))
    x = (1.0/((-k)**0.5)) * snh * u / (norm_u + 1e-8)
    z = (1.0/((-k)**0.5)) * csh
    p = torch.cat((x, z), dim=-1)
    return p

        
def spherical_projection(x: torch.Tensor, k: float):
    """
    """
    r = 1.0 / (k**0.5)
    x = r * F.normalize(x, p=2, dim=-1)
    return x
                        
                        
def lorentz_inclusion(x: torch.Tensor, k: float):      
    """
        Hyperboloid (curvature k < 0) inclusion map
    """
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
    z  = torch.sqrt(torch.clamp(norm_x**2 - (1.0/k), min=0.0))
    xz = torch.cat((x,z), 1)
    return xz



def effective_radius(k: float, eps: float):
    return (1.0 - eps) / ((-k) ** 0.5)


def project2poincare(x: torch.Tensor, k: float, eps: float=1e-3):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1.0 - eps) / ((-k) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def klein2poincare(x : torch.Tensor, k : float):
    """
        Transformation from Klein to Poincaré model

        Parameters 
        ----------
        x - Klein coordinates: torch.Tensor with shape (*, dim)
        c - Curvature in absolute value: float
            
        Output
        ---------
        y - Poincaré coordinates: torch.Tensor with shape (*, dim)
    """
    denom = 1 + torch.sqrt(1 + k * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def poincare2klein(x : torch.Tensor, k : float):
    """
        Transformation from Klein to Poincaré model

        Parameters 
        ----------
        x - Poincaré coordinates: torch.Tensor with shape (*, dim)
        k - Curvature: float 
            
        Output
        ---------
        y - Klein coordinates: torch.Tensor with shape (*, dim)
    """
    denom = 1 - k * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom



"""
    ------------------------------------------
    MOBIUS TRANSFORMS AND OPERATIONS
    ------------------------------------------
"""

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




"""
    ------------------------------------------
    CENTROIDS
    ------------------------------------------
"""

def klein_mean(x, k):
    """
        Point average in Klein model

        Parameters 
        ----------
        x - Klein coordinates: torch.Tensor with shape (*, dim)
        k - Curvature: float
            
        Output
        ---------
        mean - Klein average: torch.Tensor with shape (*, dim)
    """
    lamb = 1.0 / torch.sqrt(1 + k * x.pow(2).sum(dim=-1, keepdim=True))
    x_mean = torch.sum(lamb * x, dim=0, keepdim=False) / torch.sum(lamb, dim=0, keepdim=False)
    return x_mean


def poincare_mean(x, k):
    """
        Point average in Poincaré model

        Parameters 
        ----------
        x - Poincaré coordinates: torch.Tensor with shape (*, dim)
        c - Curvature in absolute value: float
            
        Output
        ---------
        y - Poincaré average: torch.Tensor with shape (*, dim)
    """
    y      = poincare2klein(x, k)
    y_mean = klein_mean(y, k)
    x_mean = klein2poincare(y_mean, k)
    return x_mean


def lorentz_mean(x : torch.Tensor, k : float):
    """
        Point average in Lorentz model
    """
    y = lorentz2poincare(x, k)
    y_mean = poincare_mean(y, k)
    x_mean = poincare2lorentz(y_mean, k)
    return x_mean 


def euclidean_mean(x : torch.Tensor, k : float):
    x_mean = torch.mean(x, dim=0, keepdim=False)
    return x_mean 


def mean(manifold: str, k: float):
    if manifold == 'squared_euclidean':
        return eval('lambda x : ' + 'euclidean' + '_mean(x,' + str(k) + ')')
    return eval('lambda x : ' + manifold + '_mean(x,' + str(k) + ')')
