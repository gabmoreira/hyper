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
           'train_way'        : 5,
           'train_shot'       : 1,
           'train_query'      : 15,
           'val_way'          : 5,
           'val_shot'         : 1,
           'val_query'        : 15,
           'backbone'         : 'convnet',
           'manifold'         : 'spherical',
           'manifold_dim'     : 1024,
           'manifold_k'       : 0.003,
           'metric'           : 'euclidean',
           'metric_k'         : 0.0,
           'n'                : '2'}

    init_experiment(cfg)
    
    train_samples = ImSamples(img_path=cfg['img_path'],
                              data_dict_path=cfg['train_dict_path'],
                              transforms=get_cub_transforms('train', size=84),
                              target=['class'],
                              im_padding=cfg['im_padding'],
                              preload=True)

    train_sampler = FewshotSampler(dataset=train_samples, 
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
                            transforms=get_cub_transforms('val', size=84),
                            target=['class'],
                            im_padding=cfg['im_padding'],
                            preload=True)

    val_sampler = FewshotSampler(dataset=val_samples, 
                                 num_batches=cfg['batch_size']*5,
                                 way=cfg['val_way'],
                                 shot=cfg['val_shot'],
                                 query=cfg['val_query'])
    
    val_loader = DataLoader(val_samples,
                            batch_sampler=val_sampler,
                            collate_fn=val_samples.collate_fn,
                            pin_memory=True,
                            num_workers=4)
    
    model = manifold_encoder(cfg['backbone'],
                             cfg['manifold'],
                             cfg['manifold_dim'],
                             cfg['manifold_k'],
                             cfg['riemannian'])
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                          gamma=cfg['gamma'])

    train_criterion = ProtoLoss(shot=cfg['train_shot'],
                                way=cfg['train_way'],
                                query=cfg['train_query'],
                                distance_fn=hf.cdist(cfg['metric'], cfg['metric_k']),
                                centroid_fn=hf.mean(cfg['metric'], cfg['metric_k']),
                                device=device)
    
    val_criterion = ProtoLoss(shot=cfg['val_shot'],
                              way=cfg['val_way'],
                              query=cfg['val_query'],
                              distance_fn=hf.cdist(cfg['metric'], cfg['metric_k']),
                              centroid_fn=hf.mean(cfg['metric'], cfg['metric_k']),
                              device=device)
        
    trainer = Trainer(model,
                      cfg['epochs'],
                      optimizer,
                      scheduler,
                      train_criterion,
                      val_criterion,
                      train_loader, 
                      val_loader,
                      device,
                      cfg['name'],
                      cfg['resume'])

    experiment_verbose(cfg, model, device, train_loader, val_loader)
    
    trainer.fit() 

