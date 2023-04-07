"""
    loss.py
    Mar 4 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

class ProtoLoss(nn.Module):
    def __init__(self,
                 shot:        int,
                 way:         int,
                 query:       int,
                 distance_fn: Callable,
                 centroid_fn: Callable,
                 device:      str='cuda'):
        
        super(ProtoLoss, self).__init__()
        self.shot  = shot
        self.way   = way
        self.p     = self.shot * self.way
        self.query = query
        self.distance_fn = distance_fn
        self.centroid_fn = centroid_fn
        
        self.label = torch.arange(self.way).repeat(self.query).to(device)
        
        # Store scores to compute accuracies
        self.t  = None
        self.tc = None
        
    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        x_shot, x_query = x[:self.p,...], x[self.p:,...]
        
        x_shot = x_shot.reshape((self.shot, self.way, -1))

        if self.shot > 1:
            x_prototypes = self.centroid_fn(x_shot)
        else:
            x_prototypes = x_shot.squeeze(0)
        
        logits = -self.distance_fn(x_query, x_prototypes)
        loss = F.cross_entropy(logits, self.label)
    
        self.tc = (torch.argmax(logits, dim=-1) == self.label).sum()
        self.t  = logits.shape[0]
        
        return loss
    
    def scores(self):
        return self.tc, self.t 
            
            