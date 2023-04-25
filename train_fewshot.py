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
from transforms import *

import hyperbolic.functional as hf

if __name__ == '__main__':    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
            
    cfg = {'dataset'          : 'MINI_IMAGENET',
           'img_path'         : './MINI_IMAGENET/',
           'train_dict_path'  : './MINI_IMAGENET/train_split.pt',
           'val_dict_path'    : './MINI_IMAGENET/val_split.pt',
           'test_dict_path'   : './MINI_IMAGENET/test_split.pt',
           'im_padding'       : True,
           'seed'             : 0,
           'epochs'           : 200,
           'resume'           : False,
           'batch_size'       : 100,
           'lr'               : 5e-3,
           'gamma'            : 0.8,
           'step_size'        : 60,
           'riemannian'       : False,
           'train_way'        : 30,
           'train_shot'       : 5,
           'train_query'      : 15,
           'val_way'          : 5,
           'val_shot'         : 5,
           'val_query'        : 15,
           'backbone'         : 'convnet',
           'manifold'         : 'euclidean',
           'manifold_dim'     : 1024,
           'manifold_k'       : 0.0,
           'metric'           : 'euclidean',
           'metric_k'         : 0.0,
           'n'                : '0'}

    exp_name = create_fewshot_exp_name(cfg)
    
    init_experiment(cfg, exp_name)
    
    train_samples = ImSamples(img_path=cfg['img_path'],
                              data_dict_path=cfg['train_dict_path'],
                              target=['class'],
                              preload=True)

    train_sampler = FewshotSampler(dataset=train_samples, 
                                   num_batches=cfg['batch_size'],
                                   way=cfg['train_way'],
                                   shot=cfg['train_shot'],
                                   query=cfg['train_query'])
        
    train_transforms = get_cub_transforms('train', size=84, im_padding=cfg['im_padding']),
            
    train_loader = DataLoader(train_samples,
                              batch_sampler=train_sampler,
                              collate_fn=lambda batch: train_samples.collate_fn(batch, train_transforms)
                              pin_memory=False,
                              num_workers=0)

    val_samples = ImSamples(img_path=cfg['img_path'],
                            data_dict_path=cfg['val_dict_path'],
                            target=['class'],
                            preload=True)

    val_sampler = FewshotSampler(dataset=val_samples, 
                                 num_batches=cfg['batch_size']*5,
                                 way=cfg['val_way'],
                                 shot=cfg['val_shot'],
                                 query=cfg['val_query'])
    
    val_transforms = get_cub_transforms('val', size=84, im_padding=cfg['im_padding'])
    
    val_loader = DataLoader(val_samples,
                            batch_sampler=val_sampler,
                            collate_fn=lambda batch: val_samples.collate_fn(batch, val_transforms),
                            pin_memory=False,
                            num_workers=0)
    
    model = create_manifold_encoder(cfg['backbone'],
                                    cfg['manifold'],
                                    cfg['manifold_dim'],
                                    cfg['manifold_k'],
                                    cfg['riemannian'])
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=cfg['step_size'],
                                          gamma=cfg['gamma'])

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
        
    trainer = Trainer(model,
                      cfg['epochs'],
                      optimizer,
                      scheduler,
                      train_loss,
                      val_loss,
                      train_loader, 
                      val_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    experiment_verbose(cfg, model, device, train_loader, val_loader)
    
    trainer.fit() 

