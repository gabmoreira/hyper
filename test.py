"""
    test.py
    Feb 23 2023
    Gabriel Moreira
"""

import sys
sys.path.append('./hyper')

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
import hyperbolic.functional as hf

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test few-shot experiment')
    
    parser.add_argument('dir', type=str, help='path to the experiment folder')
    parser.add_argument('shot', type=str, help='shot')
    parser.add_argument('way', type=str, help='way')
        
    args = parser.parse_args()
    
    DIR  = args.dir
    SHOT = int(args.shot)
    WAY  = int(args.way)
    
    with open(os.path.join(DIR, 'cfg.json')) as f:
        cfg = json.load(f)

    model = manifold_encoder(cfg['backbone'],
                             cfg['manifold'],
                             cfg['manifold_dim'],
                             cfg['manifold_k'],
                             cfg['riemannian']).to(device)

    model.load_state_dict(torch.load(os.path.join(DIR, 'best_weights.pt'), map_location=device))
    model.eval()

    samples = CUBData(img_path=cfg['img_path'],
                      data_dict_path=cfg['test_dict_path'],
                      transforms=get_cub_transforms(split='test'),
                      im_resize=cfg['im_resize'])

    sampler = FewshotSampler(dataset=samples, 
                             num_batches=10000,
                             way=WAY,
                             shot=SHOT,
                             query=cfg['query'])

    loader = DataLoader(samples,
                        batch_sampler=sampler,
                        collate_fn=samples.collate_fn,
                        pin_memory=True,
                        num_workers=8)

    distance_fn=hf.cdist(cfg['metric'], cfg['metric_k'])
    centroid_fn=hf.mean(cfg['metric'], cfg['metric_k'])

    criterion = ProtoLoss(shot=SHOT,
                          way=WAY,
                          query=cfg['query'],
                          distance_fn=distance_fn,
                          centroid_fn=centroid_fn)

    # Run test
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
    pprint.pprint(cfg)
    pprint.pprint("Test accuracy {:.4f} +- {:.4f}".format(m, pm))
    
    # Store results
    results = {'test_{}s{}w_mean'.format(SHOT, WAY) : m,
               'test_{}s{}w_std'.format(SHOT, WAY)  : pm}
    
    with open(os.path.join(DIR, 'results.json'), 'w') as f:
        json.dump(results, f)