"""
    test.py
    Mar 4 2023
    Gabriel Moreira
"""
import sys

import os
import json
import torch
import pprint
import numpy as np
import argparse

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import *
from loader import *
from utils  import *
from loss import *
from sampler import *
from impro import *
import hyperbolic.functional as hf


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test few-shot experiment')
    
    parser.add_argument('dir',   type=str, help='path to the experiment folder')
    parser.add_argument('shot',  type=str, help='shot')
    parser.add_argument('way',   type=str, help='way')
    parser.add_argument('query', type=str, help='query')
    
    args  = parser.parse_args()
    DIR   = str(args.dir)
    SHOT  = int(args.shot)
    WAY   = int(args.way)
    QUERY = int(args.query)
    
    with open(os.path.join(DIR, 'cfg.json')) as f:
        cfg = json.load(f)

    model = create_manifold_encoder(cfg['backbone'],
                                    cfg['manifold'],
                                    cfg['manifold_dim'],
                                    cfg['manifold_k'],
                                    cfg['riemannian'],
                                    cfg['clip'] if 'clip' in cfg.keys() else None)

    model.load_state_dict(torch.load(os.path.join(DIR, 'best_weights.pt'),
                                     map_location=device))
    model.eval()
    model = model.to(device)
    
    samples = ImSamples(img_path=cfg['img_path'],
                        data_dict_path=cfg['test_dict_path'],
                        target=['class'],
                        transforms=get_cub_transforms('test', size=84))

    sampler = FewshotSampler(targets=samples.data['target'], 
                             num_batches=10000,
                             way=WAY,
                             shot=SHOT,
                             query=QUERY)
        
    loader = DataLoader(samples,
                        batch_sampler=sampler,
                        collate_fn=samples.collate_fn,
                        pin_memory=True,
                        num_workers=8)

    distance_fn=hf.cdist(cfg['metric'], cfg['metric_k'])
    centroid_fn=hf.mean(cfg['metric'], cfg['metric_k'])

    criterion = ProtoLoss(shot=SHOT,
                        way=WAY,
                        query=QUERY,
                        distance_fn=distance_fn,
                        centroid_fn=centroid_fn,
                        device=device)

    test_acc_record = np.zeros((10000,))

    num_correct = 0
    num_trials  = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            data  = batch['data'].to(device)
            x     = model(data)
            loss  = criterion(x)
            tc, t = criterion.scores()

            num_correct += tc
            num_trials += t
            test_acc_record[i] = tc / t

    m   = np.mean(test_acc_record)
    std = np.std(test_acc_record)
    pm  = 1.96 * (std / np.sqrt(len(test_acc_record)))
    print("Test accuracy {:.4f} +- {:.4f}".format(m, pm))
    
    # Store results
    results = {'test_{}s{}w_mean'.format(SHOT, WAY) : m,
               'test_{}s{}w_std'.format(SHOT, WAY)  : pm}
    
    if os.path.exists(os.path.join(DIR, 'results.json')):
        print('Results file already exists. Appending to dictionary.')
        with open(os.path.join(DIR, 'results.json')) as f:
            results = json.load(f)
        results['test_{}s{}w_mean'.format(SHOT, WAY)] = m
        results['test_{}s{}w_std'.format(SHOT, WAY)]  = pm
    else:
        print('Results file does not exist. Creating dictionary.')
        results = {'test_{}s{}w_mean'.format(SHOT, WAY) : m,
                   'test_{}s{}w_std'.format(SHOT, WAY)  : pm}
        
    with open(os.path.join(DIR, 'results.json'), 'w') as f:
        json.dump(results, f)