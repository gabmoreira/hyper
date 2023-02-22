"""
    train.py
    Oct 13 2022
    Gabriel Moreira
"""

import torch
import torch.nn as nn


class HyperDistance(nn.Module):
    def __init__(self, metric):
        """
        Batch-mean hyperbolic distortion 
        
        Parameters 
        ----------
        metric - torch.tensor of shape n with hyperbolic metric signature
        """
        super(HyperDistance, self).__init__()
        self.metric = metric
        
    def forward(self, batch_features, target_features):
        inner_products = torch.diag(torch.matmul(self.metric * batch_features, target_features.transpose(1,0)))
        distances = torch.acosh(torch.clamp(-inner_products, min=1.0))
        loss = torch.mean(distances)
        
        return loss


    
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
        
            
            