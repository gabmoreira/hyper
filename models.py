"""
    models.py
    Sep 8 2022
    Gabriel Moreira
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

class MLPClassifier(nn.Module):
    
    def __init__(self, dim_in, feat_dim, num_classes):
        super(MLPClassifier, self).__init__()
        
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.fc2 = nn.Linear(dim_in, feat_dim)
        
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.classifier(x)
        
        return x



class Resnet50Feat(nn.Module):
    def __init__(self, unfrozen_layers):
        """
        unfrozen_layers should be list e.g., ['layer3', 'layer4']
        """
        super(Resnet50Feat, self).__init__()
        
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
        
        
        
        
        
        
        
class Resnet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50Classifier, self).__init__()
        
        self.backbone   = Resnet50Feat(unfrozen_layers=['layer3', 'layer4'])
        self.classifier = MLPClassifier(2048, feat_dim=128, num_classes=num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    
    
    
class HyperbolicFeat(nn.Module):
    def __init__(self):
        super(HyperbolicFeat, self).__init__()
        self.backbone = Resnet50Feat(unfrozen_layers=['layer3', 'layer4'])
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,  127)
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) # local coordinates
        
        z = torch.sqrt(torch.square(torch.norm(x, p=2, dim=1, keepdim=True)) + 1.0)
        
        hyperbolic_features = torch.cat((x,z), 1)
        
        return hyperbolic_features