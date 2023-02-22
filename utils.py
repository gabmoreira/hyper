"""
    utils.py
    Feb 21 2023
    Gabriel Moreira
"""

import os
import json


def save_config(exp_name, cfg):
    with open(exp_name + '/cfg.json', 'w') as fp:
        json.dump(cfg, fp)
     
    

def init_experiment(cfg):
    exp_name = cfg['dataset'] + '_' +\
               cfg['manifold'] + cfg['manifold_dim'] + '_' \ 
               cfg['metric'] + '_n' + cfg['n']
    
    cfg['name'] = exp_name
    
    # If experiment folder doesn't exist create it
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
        print("Created experiment folder : ", exp_name)
    else:
        print(exp_name, "folder already exists.")

    save_config(exp_name, cfg)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])


    
def experiment_verbose(cfg, device, train_loder, val_loader):
    print(model)
    verbose = 'Experiment ' + cfg['name'] + '\n' + \
              'Running on' + str(device) + '\n' \
              'Train - {} batches of size {}'.format(len(train_loader),
                                                     cfg['batch_size']) + '\n' +\
              'Val   - {} batches of size {}'.format(len(val_loader),
                                                     cfg['batch_size'])
    print(verbose)