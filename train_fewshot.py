"""
    train.py
    Mar 4 2023
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
from impro import *

import hyperbolic.functional as hf

if __name__ == '__main__':    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
            
    cfg = {'dataset'         : 'MINI_IMAGENET',
           'img_path'        : './MINI_IMAGENET/',
           'train_dict_path' : './MINI_IMAGENET/train_split.pt',
           'val_dict_path'   : './MINI_IMAGENET/val_split.pt',
           'test_dict_path'  : './MINI_IMAGENET/test_split.pt',
           'seed'            : 10,
           'epochs'          : 200,
           'resume'          : False,
           'batch_size'      : 100,
           'lr'              : 5e-3,
           'gamma'           : 0.8,
           'step_size'       : 60,
           'riemannian'      : False,
           'train_way'       : 20,
           'train_shot'      : 5,
           'train_query'     : 15,
           'val_way'         : 5,
           'val_shot'        : 5,
           'val_query'       : 15,
           'backbone'        : 'convnet',
           'manifold'        : 'euclidean',
           'manifold_dim'    : 128,
           'manifold_k'      : 0.0,
           'metric'          : 'squared_euclidean',
           'metric_k'        : 0.0,
           'clip'            : None,
           'n'               : 'Adam_17.06'}

    exp_name = create_fewshot_exp_name(cfg)
    
    init_experiment(cfg, exp_name)
    
    train_samples = ImSamples(img_path=cfg['img_path'],
                              data_dict_path=cfg['train_dict_path'],
                              target=['class'],
                              preload=False,
                              transforms=get_cub_transforms('train', size=84))

    train_sampler = FewshotSampler(targets=train_samples.data['target'], 
                                   num_batches=cfg['batch_size'],
                                   way=cfg['train_way'],
                                   shot=cfg['train_shot'],
                                   query=cfg['train_query'])
        
    train_loader = DataLoader(train_samples,
                              batch_sampler=train_sampler,
                              collate_fn=train_samples.collate_fn,
                              pin_memory=True,
                              num_workers=4)

    val_samples = ImSamples(img_path=cfg['img_path'],
                            data_dict_path=cfg['val_dict_path'],
                            target=['class'],
                            preload=False,
                            transforms=get_cub_transforms('val', size=84))

    val_sampler = FewshotSampler(targets=val_samples.data['target'], 
                                 num_batches=cfg['batch_size']*5,
                                 way=cfg['val_way'],
                                 shot=cfg['val_shot'],
                                 query=cfg['val_query'])
        
    val_loader = DataLoader(val_samples,
                            batch_sampler=val_sampler,
                            collate_fn=val_samples.collate_fn,
                            pin_memory=True,
                            num_workers=4)
    
    model = create_manifold_encoder(cfg['backbone'],
                                    cfg['manifold'],
                                    cfg['manifold_dim'],
                                    cfg['manifold_k'],
                                    cfg['riemannian'],
                                    cfg['clip'])
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])

    train_loss = ProtoLoss(shot=cfg['train_shot'],
                           way=cfg['train_way'],
                           query=cfg['train_query'],
                           distance_fn=hf.cdist(cfg['metric'], cfg['metric_k']),
                           centroid_fn=hf.mean(cfg['metric'], cfg['metric_k']),
                           device=device)
    
    val_loss = ProtoLoss(shot=cfg['val_shot'],
                         way=cfg['val_way'],
                         query=cfg['val_query'],
                         distance_fn=hf.cdist(cfg['metric'], cfg['metric_k']),
                         centroid_fn=hf.mean(cfg['metric'], cfg['metric_k']),
                         device=device)
        
    trainer = Trainer(model=model,
                      epochs=cfg['epochs'],
                      optimizer=optimizer,
                      scheduler=scheduler,
                      train_loss=train_loss,
                      val_loss=val_loss,
                      train_loader=train_loader, 
                      val_loader=val_loader,
                      val_freq=10,
                      best_on='val_acc',
                      device=device,
                      name=cfg['name'],
                      resume=cfg['resume'])

    experiment_verbose(cfg, model, device, train_loader, val_loader)
    
    trainer.fit() 

