"""
    train.py
    Oct 13 2022
    Gabriel Moreira
"""

import os
import json
import numpy as np

from time import gmtime, strftime

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T

from loader import *
from torch.utils.data import DataLoader

from models import *

from trainer import Trainer

from loss import HyperDistance


def save_config(cfg):
    with open(cfg['name'] + '/cfg.json', 'w') as fp:
        json.dump(cfg, fp)
     
    
def create_experiment_name():
    name = strftime("%Y%m%d%H%M%S", gmtime())
    return name


def init_experiment(cfg):
    # If experiment folder doesn't exist create it
    if not os.path.isdir(cfg['name']):
        os.makedirs(cfg['name'])
        print("Created experiment folder : ", cfg['name'])
    else:
        print(cfg['name'], "folder already exists.")

    save_config(cfg)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])


        
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        
    # Configuration params
    cfg = {'img_path'            : '../ifetch/deepfashion/in_shop',
           'train_dict_path'     : './preprocessed/train_split.pt',
           'val_dict_path'       : './preprocessed/val_split.pt',
           'taxonomy_path'       : './preprocessed/taxonomy_extended_hyperbolic.pt',
           'seed'                : 10,
           'epochs'              : 120,
           'batch_size'          : 1024,
           'lr'                  : 1e-1,
           'resume'              : False,
           'name'                : 'new'}

    
    data_T = {'train' : T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.)),
                                   T.RandomHorizontalFlip(),
                                   T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                   T.RandomGrayscale(p=0.2),
                                   T.ToTensor(),
                                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
              'val' : T.Compose([T.ToTensor(),
                                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    init_experiment(cfg)

    train_samples = DeepFashionData(img_path=cfg['img_path'],
                                    data_dict_path=cfg['train_dict_path'],
                                    taxonomy_path=cfg['taxonomy_path'],
                                    transforms=data_T['train'])

    train_loader = DataLoader(train_samples,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              collate_fn=train_samples.collate_fn)

    val_samples = DeepFashionData(img_path=cfg['img_path'],
                                  data_dict_path=cfg['val_dict_path'],
                                  taxonomy_path=cfg['taxonomy_path'],
                                  transforms=data_T['val'])

    val_loader = DataLoader(dev_samples,
                            batch_size=cfg['batch_size'],
                            shuffle=False,
                            collate_fn=val_samples.collate_fn)
    
    model = HyperbolicFeat()
    model = model.to(device)
    "model.load_state_dict(torch.load(''))

    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='max')
    
    metric = torch.ones(128, device=device, dtype=torch.float32)
    metric[-1] = -1.0
    criterion = HyperDistance(metric=metric) #nn.CrossEntropyLoss(reduction='mean')

    trainer = Trainer(model,
                      cfg['epochs'],
                      optimizer,
                      scheduler,
                      criterion,
                      train_loader, 
                      dev_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    print('Experiment ' + cfg['name'])
    print('Running on', device)
    print('Train - {} batches of size {}'.format(len(train_loader), cfg['batch_size']))
    print('  Val - {} batches of size {}'.format(len(dev_loader), cfg['batch_size']))
    print(model)

    trainer.fit() 

