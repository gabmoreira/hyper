"""
    sampler.py
    Feb 22 2023
    Gabriel Moreira
"""

import torch
import numpy as np


class FewshotSampler:
    def __init__(self, 
                 dataset,
                 num_batches: int, 
                 way: int,
                 shot: int,
                 query: int):
        
        self.num_batches = num_batches
        self.way   = way
        self.shot  = shot
        self.query = query
        self.samples_per_class = shot + query

        self.classes = np.array(dataset.data['target'])
        self.unique_classes = np.unique(self.classes)
        self.num_classes = len(self.unique_classes)
        
        self.idx = []
        for cls in self.unique_classes:
            i = np.argwhere(cls == self.classes).reshape(-1)
            i = torch.from_numpy(i)
            self.idx.append(i)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i_batch in range(self.num_batches):
            batch = []
            # pick #way classes
            classes = torch.randperm(len(self.idx))[: self.way]
            
            for cls in classes:
                cls_idx = self.idx[cls]
                pos = torch.randperm(len(cls_idx))[: self.samples_per_class]
                batch.append(cls_idx[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch