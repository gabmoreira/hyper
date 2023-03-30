"""
    backbone.py
    Mar 4 2023
    Gabriel Moreira
"""

import torch
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights


class Resnet50(nn.Module):
    def __init__(self, unfrozen_layers: list):
        """
            Unfrozen_layers should be e.g., ['layer3', 'layer4']
        """
        super(Resnet50, self).__init__()
        
        self.backbone    = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
        self.backbone.fc = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for layer in unfrozen_layers:
            for param in getattr(self.backbone, layer).parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    
    
def conv_block(in_dim: int, out_dim: int):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, padding=1),
                         nn.BatchNorm2d(out_dim),
                         nn.ReLU(),
                         nn.MaxPool2d(2))


class Convnet(nn.Module):
    def __init__(self, in_dim=3, hid_dim=64, out_dim=64):
        """
        """
        super().__init__()
        self.encoder = nn.Sequential(conv_block(in_dim, hid_dim),
                                     conv_block(hid_dim, hid_dim),
                                     conv_block(hid_dim, hid_dim),
                                     conv_block(hid_dim, out_dim))

    def forward(self, x):
        """
        """
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x