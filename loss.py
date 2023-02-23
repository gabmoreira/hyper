"""
    train.py
    Oct 13 2022
    Gabriel Moreira
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

class ProtoLoss(nn.Module):
    def __init__(self,
                 shot: int,
                 way: int,
                 query: int,
                 distance_fn: Callable,
                 centroid_fn: Callable,
                 device='cuda'):
        
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
        
    def forward(self, x: torch.Tensor, target):
        x_shot, x_query = x[:self.p,...], x[self.p:,...]
        
        x_shot = x_shot.reshape((self.shot, self.way, -1))

        if self.shot > 1:
            x_prototypes = self.centroid_fn(x_shot, dim=0)
        else:
            x_prototypes = x_shot.squeeze(0)
        
        logits = -self.distance_fn(x_query, x_prototypes)
        loss = F.cross_entropy(logits, self.label)
    
        self.tc = (torch.argmax(logits, dim=-1) == self.label).sum()
        self.t  = logits.shape[0]
    
    def scores(self):
        return self.tc, self.t 

    
class NNAcc:
    def __init__(self, node_embeddings, voc, metric):
        """
        Nearest-Neighbor Classification Accuracy
        
        Parameters 
        ----------
        node_embeddings - dictionary with full node name as key
        and node embedding as value. E.g., node_embeddings['WOMEN/Skirts']=torch.tensor(...)
        voc             - vocabulary dictionary 
        metric          - torch.tensor  of shape n with hyperbolic metric signature
        """
        super(NNAcc, self).__init__()
        
        node_labels = list(node_embeddings.keys())
        
        self.metric = metric
        self.voc    = voc
        
        self.cat_labels = []
        for label in node_labels:
            if len(label.split('/')) == 2:
                self.cat_labels.append(label)
        
        self.cat_embeddings = torch.stack([node_embeddings[label] for label in self.cat_labels], dim=0)
        
        
    def acc(self, features, cat):
        inner_products = torch.matmul(self.metric * features, self.cat_embeddings.transpose(1,0))
        distances      = torch.acosh(-inner_products)
        
        predictions = torch.argmin(distances, dim=1, keepdim=False)
        
        total = 0
        for i in range(len(predictions)):
            pred_cat = self.cat_labels[predictions[i]].split('/')[-1]
            if self.voc['cat']._idx2word[cat[i].item()] == pred_cat:
                total += 1
                
        return total
    
    
    def __call__(self, features, cat):
        return self.acc(features, cat)
        
            
            