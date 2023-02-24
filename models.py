"""
    models.py
    Feb 22 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperbolic.nn as hnn
import hyperbolic.functional as hf

from backbone import Resnet50, Convnet
                        
                                           
def manifold_encoder(backbone: str,
                     manifold: str,
                     dim: int,
                     k: float,
                     riemannian: bool):

    if backbone == 'resnet50':
        conv = nn.Sequential(Resnet50(unfrozen_layers=['layer2', 
                                                       'layer3',
                                                       'layer4']),
                             nn.Linear(2048, dim))                                    
    elif backbone == 'convnet':
        conv = Convnet(out_dim=dim)
                              
    if manifold.lower() == 'poincare':
        assert k < 0
        to_manifold = hnn.PoincareExp0(k, riemannian)
    elif manifold.lower() == 'spherical':
        assert k > 0
        to_manifold = hnn.SphericalProjection(k)
    elif manifold.lower() == 'euclidean':
        assert k == 0
        to_manifold = nn.Identity()
                                               
    model = nn.Sequential(conv, to_manifold)
                                           
    return model