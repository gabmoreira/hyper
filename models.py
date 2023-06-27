"""
    models.py
    Mar 12 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperbolic.nn as hnn
import hyperbolic.functional as hf

from backbone import Convnet
                        

def create_mlp(in_dim: int, mlp_dims: str):
    """
        Create MLP with mlp_dims e.g., "128-128-128"
    """
    mlp_spec = f"{in_dim}-{mlp_dims}"
    layers = []
    
    f = list(map(int, mlp_spec.split("-")))
    
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    
    return nn.Sequential(*layers)


    
def create_manifold_encoder(backbone: str,
                            manifold: str,
                            dim: int,
                            k: float,
                            riemannian: bool,
                            clip: float=None):
                      
    if backbone == 'convnet':
        conv = Convnet(out_dim=dim)
                              
    if manifold.lower() == 'poincare':
        assert k < 0
        to_manifold = hnn.PoincareExp0(k, riemannian, clip)
        
    if manifold.lower() == 'lorentz':
        assert k < 0
        to_manifold = hnn.LorentzExp0(k)
        
    elif manifold.lower() == 'spherical':
        assert k > 0
        to_manifold = hnn.SphericalProjection(k)
        
    elif manifold.lower() == 'euclidean':
        assert k == 0
        to_manifold = nn.Identity()
     
    model = nn.Sequential(conv, to_manifold)
                               
    return model