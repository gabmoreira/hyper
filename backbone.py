import torch
from torch import nn
import torch.nn.functional as F



class Resnet50(nn.Module):
    def __init__(self, unfrozen_layers):
        """
        unfrozen_layers should be list e.g., ['layer3', 'layer4']
        """
        super(Resnet50, self).__init__()
        
        self.backbone    = resnet50(pretrained=True) 
        self.backbone.fc = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for layer in unfrozen_layers:
            for param in getattr(self.backbone, layer).parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        return x