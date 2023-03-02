"""
    train.py
    Feb 21 2023
    Gabriel Moreira
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from loader import *
from loss import *
from models import *
from trainer import Trainer
from utils import *
from sampler import *


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
            
    cfg = {'dataset'          : 'CUB_200_2011',
           'img_path'         : './CUB_200_2011/',
           'train_dict_path'  : './CUB_200_2011/train_split.pt',
           'val_dict_path'    : './CUB_200_2011/val_split.pt',
           'test_dict_path'   : './CUB_200_2011/test_split.pt',
           'im_padding'       : True,
           'seed'             : 10,
           'epochs'           : 200,
           'resume'           : False,
           'batch_size'       : 100,
           'lr'               : 1e-3,
           'gamma'            : 0.8,
           'step_size'        : 40,
           'riemannian'       : False,
           'way'              : 5,
           'shot'             : 5,
           'query'            : 15,
           'backbone'         : 'convnet',
           'manifold'         : 'lorentz',
           'manifold_dim'     : 1024,
           'manifold_k'       : -0.05,
           'metric'           : 'lorentz',
           'metric_k'         : -0.05,
           'n'                : 0}

    init_experiment(cfg)
    
    train_samples = ImSamples(img_path=cfg['img_path'],
                              data_dict_path=cfg['train_dict_path'],
                              transforms=get_cub_transforms('train'),
                              im_padding=cfg['im_padding'])

    train_sampler = FewshotSampler(dataset=train_samples, 
                                   num_batches=cfg['batch_size'],
                                   way=cfg['way'],
                                   shot=cfg['shot'],
                                   query=cfg['query'])
        
    train_loader = DataLoader(train_samples,
                              batch_sampler=train_sampler,
                              collate_fn=train_samples.collate_fn,
                              num_workers=8,
                              pin_memory=True)

    val_samples = ImSamples(img_path=cfg['img_path'],
                            data_dict_path=cfg['val_dict_path'],
                            transforms=get_cub_transforms('val'),
                            im_padding=cfg['im_padding'])

    val_sampler = FewshotSampler(dataset=val_samples, 
                                 num_batches=cfg['batch_size']*5,
                                 way=cfg['way'],
                                 shot=cfg['shot'],
                                 query=cfg['query'])
    
    val_loader = DataLoader(val_samples,
                            batch_sampler=val_sampler,
                            collate_fn=val_samples.collate_fn,
                            num_workers=8,
                            pin_memory=True)
    
    model = manifold_encoder(cfg['backbone'],
                             cfg['manifold'],
                             cfg['manifold_dim'],
                             cfg['manifold_k'],
                             cfg['riemannian'])
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                          gamma=cfg['gamma'])

    criterion = ProtoLoss(shot=cfg['shot'],
                          way=cfg['way'],
                          query=cfg['query'],
                          distance_fn=hf.cdist(cfg['metric'], cfg['metric_k']),
                          centroid_fn=hf.mean(cfg['metric'], cfg['metric_k']))
    
    trainer = Trainer(model,
                      cfg['epochs'],
                      optimizer,
                      scheduler,
                      criterion,
                      train_loader, 
                      val_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    experiment_verbose(cfg, model, device, train_loader, val_loader)
    
    trainer.fit() 

