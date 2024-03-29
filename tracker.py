"""
    tracker.py
    Mar 4 2023
    Gabriel Moreira
"""
import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

class Tracker:
    def __init__(self,
                 filename: str,
                 load: bool=False):

        """
        """
        self.filename = os.path.join(filename, 'tracker.csv')

        if load:
            self.metrics_dict = self.load()
            self.init = False
        else:        
            self.metrics_dict = {}
            self.init = True
        
        
    def update(self, **kwargs):
        """
        """
        if self.init:
            for metric, value in kwargs.items():
                self.metrics_dict[metric] = [value]
            self.init = False
        else:
            for metric, value in kwargs.items():
                assert metric in self.metrics_dict.keys(), 'unknown metric'
                self.metrics_dict[metric].append(value)

        self.save()


    def is_larger(self, metric: str, value: float):
        """
        """
        if self.init:
            return True
        
        assert metric in self.metrics_dict.keys(), 'unknown metric'
        if len(self.metrics_dict[metric]) > 0:
            return max(self.metrics_dict[metric]) <= value
        else:
            return True


    def is_smaller(self, metric: str, value: float):
        """
        """
        if self.init:
            return True
        
        assert metric in self.metrics_dict.keys(), 'unknown metric'
        if len(self.metrics_dict[metric]) > 0:
            return min(self.metrics_dict[metric]) >= value
        else:
            return True


    def is_better(self, metric: str, value: float):
        """
        """
        if self.init:
            return True
        
        assert(metric in self.metrics_dict.keys())
        if len(self.metrics_dict[metric]) > 0:
            if metric == 'val_acc':
                return self.is_larger(metric, value)
            elif metric == 'val_loss':
                return self.is_smaller(metric, value)
        else:
            return True
        
        
    def save(self):
        """
        """
        df = pd.DataFrame.from_dict(self.metrics_dict)
        df = df.set_index('epoch')
        df.to_csv(self.filename)


    def load(self):
        """
        """
        df = pd.read_csv(self.filename)  
        metrics_dict = df.to_dict(orient='list')
        return metrics_dict


    def __len__(self):
        """
        """
        return len(self.metrics_dict)
    
    

    
def load_tracker_data(dataset, experiments_dir, selection=None):
    """
    """
    experiments = [e for e in os.listdir(experiments_dir) if e.split('_')[0].lower() == dataset.lower()]
    
    if selection is not None:
        experiments = [e for e in experiments if e in selection]
        
    trackers = {}
    cfgs     = {}
    for experiment in experiments:
        tracker_path = os.path.join(experiments_dir, experiment, 'tracker.csv')
        cfg_path     = os.path.join(experiments_dir, experiment, 'cfg.json')
        trackers[experiment] = pd.read_csv(tracker_path)
        with open(cfg_path) as f:
            data = json.load(f)
            cfgs[experiment] = data

    print(experiments)
    return trackers, cfgs


def plot_trackers(trackers, cfgs):
    """
    """
    plt.figure(figsize=(18,13))
    sns.set_theme()
    plt.subplot(2,2,1)
    for k, tracker in trackers.items():
        sns.lineplot(data=tracker, y="val_acc", x='epoch', linewidth=0.6)    
    plt.legend(labels=['{}  |  {}'.format(cfgs[k]['name'], cfgs[k]['backbone']) 
                       for k in cfgs.keys()], fontsize=7)

    plt.subplot(2,2,2)
    for k, tracker in trackers.items():
        sns.lineplot(data=tracker, y="train_acc", x='epoch', linewidth=0.6)    
    plt.legend(labels=['{}  |  {}'.format(cfgs[k]['name'], cfgs[k]['backbone'])  
                       for k in cfgs.keys()], fontsize=7)

    plt.subplot(2,2,3)
    for k, tracker in trackers.items():
        sns.lineplot(data=tracker, y="val_loss", x='epoch', linewidth=0.6)    
    plt.legend(labels=['{}  |  {}'.format(cfgs[k]['name'], cfgs[k]['backbone']) 
                       for k in cfgs.keys()], fontsize=7)

    plt.subplot(2,2,4)
    for k, tracker in trackers.items():
        sns.lineplot(data=tracker, y="train_loss", x='epoch', linewidth=0.6)    
    plt.legend(labels=['{}  |  {}'.format(cfgs[k]['name'], cfgs[k]['backbone']) 
                       for k in cfgs.keys()], fontsize=7)