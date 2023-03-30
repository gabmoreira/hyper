"""
    hyperbolic/functional.py
    Mar 30 2023
    Gabriel Moreira
"""
import torch
import numpy as np
from tqdm.auto import tqdm 

from typing import Callable

def delta_hyp(cdist: torch.Tensor):
    """
        Computes delta hyperbolicity value from distance matrix
    """

    p = 0
    row   = cdist[p, :].unsqueeze(0)
    col   = cdist[:, p].unsqueeze(-1)
    gprod = 0.5 * (row + col - cdist)

    maxmin = torch.max(torch.minimum(gprod[:, :, None], gprod[None, :, :]), axis=1)[0]
    delta  = torch.max(maxmin - gprod)
    return delta


def delta_hyperbolicity(x : torch.Tensor,
                        distance_fn: Callable,
                        maxiter : int=10,
                        batch_size : int =1500):
    """
        Computes delta hyperbolicity in batches from tensor of points
    """
    vals = []
    for _ in tqdm(range(maxiter)):
        idx = np.random.randint(0, x.shape[0], size=batch_size)
        x_batch = x[idx,...]
        cdist   = distance_fn(x_batch)
        diam    = cdist.max()
        delta_rel = 2 * delta_hyp(cdist) / diam
        
        vals.append(delta_rel)
        
    return max(vals)