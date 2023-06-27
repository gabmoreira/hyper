"""
    loss.py
    Mar 4 2023
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


class VICLoss(nn.Module):
    def __init__(self,
                 manifold_dim: int,
                 sim_coef: float,
                 std_coef: float,
                 cov_coef: float):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef
        
    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor=None):
        x = x.view(-1, 2, self.manifold_dim)

        x1 = x[:,0,...]
        x2 = x[:,1,...]

        repr_loss = F.mse_loss(x1, x2)
        
        x1 = x1 - x1.mean(dim=0)
        x2 = x2 - x2.mean(dim=0)
        std_x1 = torch.sqrt(x1.var(dim=0) + 0.0001)
        std_x2 = torch.sqrt(x2.var(dim=0) + 0.0001)
        
        std_loss = torch.mean(F.relu(1 - std_x1)) / 2 + torch.mean(F.relu(1 - std_x2)) / 2

        cov_x1 = (x1.T @ x1) / (x1.shape[0] - 1)
        cov_x2 = (x2.T @ x2) / (x2.shape[0] - 1)
        
        n = cov_x1.shape[0]
        
        cov_loss_x1 = cov_x1.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten().pow_(2).sum().div(self.manifold_dim)
        cov_loss_x2 = cov_x2.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten().pow_(2).sum().div(self.manifold_dim)

        loss = self.sim_coef * repr_loss
        loss += self.std_coef * std_loss
        loss += self.cov_coef * (cov_loss_x1 + cov_loss_x2)
        
        return loss
    
    
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
    
        self.tc = (torch.argmax(logits, dim=-1) == self.label).sum().detach().cpu().item()
        self.t  = logits.shape[0]
        
        return loss
    
    def scores(self):
        return self.tc, self.t 
            
            