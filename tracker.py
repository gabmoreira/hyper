'''
    File name: utils.py
    Author: Gabriel Moreira
    Date last modified: 03/08/2022
    Python Version: 3.7.10
'''

import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

class Tracker:
    def __init__(self, metrics, filename, load=False):
        '''
        '''
        self.filename = os.path.join(filename, 'tracker.csv')

        if load:
            self.metrics_dict = self.load()
        else:        
            self.metrics_dict = {}
            for metric in metrics:
                self.metrics_dict[metric] = []


    def update(self, **args):
        '''
        '''
        for metric in args.keys():
            assert(metric in self.metrics_dict.keys())
            self.metrics_dict[metric].append(args[metric])

        self.save()


    def isLarger(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[-1] < value


    def isSmaller(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[0] > value


    def save(self):
        '''
        '''
        df = pd.DataFrame.from_dict(self.metrics_dict)
        df = df.set_index('epoch')
        df.to_csv(self.filename)


    def load(self):
        '''
        '''
        df = pd.read_csv(self.filename)  
        metrics_dict = df.to_dict(orient='list')
        return metrics_dict


    def __len__(self):
        '''
        '''
        return len(self.metrics_dict)