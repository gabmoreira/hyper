"""
    utils.py
    Feb 21 2023
    Gabriel Moreira
"""

import os
import json
import torch
import numpy as np
import pprint

def save_config(exp_name, cfg):
    with open(exp_name + '/cfg.json', 'w') as fp:
        json.dump(cfg, fp)
     
    
def init_experiment(cfg):
    exp_name = cfg['dataset'] + '_'
    exp_name += cfg['manifold'] + str(cfg['manifold_dim']) + '_'
    exp_name += cfg['metric'] + '_' + str(cfg['shot']) + 's' + str(cfg['way']) + 'w_n' + str(cfg['n'])
    
    cfg['name'] = exp_name
    
    # If experiment folder doesn't exist create it
    if not os.path.isdir(os.path.join('./experiments', exp_name)):
        os.makedirs(os.path.join('./experiments', exp_name))
        print("Created experiment folder : ", exp_name)
    else:
        print(exp_name, "folder already exists.")

    with open(os.path.join('./experiments', exp_name, 'cfg.json'), 'w') as f:
        json.dump(cfg, f)
        
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])


    
def experiment_verbose(cfg, model, device, train_loader, val_loader):
    verbose = 'Experiment: ' + cfg['name'] + '\n'
    verbose += 'Running on ' + str(device) + '\n'
    verbose += 'Train - {} batches of size {}'.format(len(train_loader), cfg['batch_size']) + '\n'
    verbose +=  'Val   - {} batches of size {}'.format(len(val_loader), cfg['batch_size'])
    print('\n')
    pprint.pprint(cfg)
    print('\n')
    print(verbose)