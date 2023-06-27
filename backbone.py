"""
    backbone.py
    Mar 4 2023
    Gabriel Moreira
"""

import torch
from torch import nn



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
    
        
def conv_block(in_dim: int, out_dim: int):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, padding=1),
                         nn.BatchNorm2d(out_dim),
                         nn.ReLU(),
                         nn.MaxPool2d(2))




class ResidualMerge(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        x = x + self.model(x)
        return x


class ResNLH(nn.Module):
    def __init__(self,
                 n: int,
                 l: int,
                 h: int,
                 out_dim: int,
                 in_dim: int=3):
        super().__init__()
        
        in_layers = [nn.Conv2d(in_dim, h, 2, stride=2),
                     nn.BatchNorm2d(h),
                     nn.ReLU()]
        
        for _ in range(n-1):
            in_layers.extend([nn.Conv2d(h, h, 2, stride=2),
                              nn.BatchNorm2d(h),
                              nn.ReLU()])
        
        self.conv_block = nn.Sequential(*in_layers)
        
        residual_blocks = []
        for _ in range(l):
            residual_blocks.extend([ResidualMerge(nn.Sequential(nn.Conv2d(h, h, 3, stride=1, padding=1),
                                                  nn.BatchNorm2d(h),
                                                  nn.ReLU(),
                                                  nn.Conv2d(h, h, 3, stride=1, padding=1),
                                                  nn.BatchNorm2d(h))),
                                    nn.ReLU()])
            
        self.residuals = nn.Sequential(*residual_blocks)
        self.flatten   = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc        = nn.Linear(6400, out_dim)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.residuals(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x