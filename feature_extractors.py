"""
    feature_extractors.py
    Sep 8 2022
    Gabriel Moreira
"""
import torch
import torch.nn as nn

import clip
from torchvision.models import resnet50

class FeatureExtractor(nn.Module):
    def __init__(self, arch):
        super(FeatureExtractor, self).__init__()
        
        assert arch in ['resnet50', 'vitb32']
        
        self.arch = arch
        
        self.feat_dim           = None
        self.normalization_mean = None
        self.normalization_std  = None
        
        if self.arch == 'resnet50':
            self.feat_dim           = 2048
            self.normalization_mean = [0.485, 0.456, 0.406]
            self.normalization_std  = [0.229, 0.224, 0.225]
            self.backbone = resnet50(pretrained=True)   
            self.backbone.fc = nn.Identity() # Remove last layer
            
        elif self.arch == 'vitb32':
            self.feat_dim           = 512
            self.normalization_mean = [0.48145466, 0.4578275, 0.40821073]
            self.normalization_std  = [0.26862954, 0.26130258, 0.27577711]
            clip_model, _ = clip.load("ViT-B/32", device='cuda')
            self.backbone = clip_model.visual # Vision Transformer from CLIP
            
    def forward(self, x):
        if self.arch == 'vitb32':
            x = x.to(torch.float16)
            
        x = self.backbone(x)
        return x