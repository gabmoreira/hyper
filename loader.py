"""
    loader.py
    Mar 4 2023
    Gabriel Moreira
"""
import os
import math
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from typing import List

from voc import Vocabulary


    
class ImSamples(Dataset):
    def __init__(self,
                 img_path: str,
                 data_dict_path: str=None,
                 target: List[str]=None,
                 preload: bool=False,
                 transforms = None):
 
        self.img_path   = img_path
        self.preload    = preload
        self.target     = target
        self.transforms = transforms

        """
            Reads dict with all data from data_dict_path

            Examples:
                dict['path']       = ['images/159.Black_and_white_Warbler/Black_...jpg', ...]    
                dict['attributes'] = [[1 0 22 3 0 4 4 0 ...], ... ]
                dict['class']      = ['159.Black_and_white_Warbler', ... ]
                dict['bbox']       = [[50, 40, 140, 180], [10, 55, 78, 180], ...]
        """
        if data_dict_path is not None:
            self.data = {}
            data_dict = torch.load(data_dict_path)
            for key, value in data_dict.items():
                self.data[key] = value
        else:
            files = os.listdir(os.path.join(self.img_path, 'images'))
            self.data = {'path' : ['images/' + f for f in files]}
            
        if target is not None:
            self.build_target('target', target)
            self.build_vocs(['target'])
        
        self.length = len(self.data['path'])
        
        if self.preload:
            self.load()
            
        self.verbose()
        
        
    def build_vocs(self, items: list):
        """
            Creates a dictionary of vocabularies for the items requested
        """
        self.voc = {item : Vocabulary(self.data[item]) for item in items}

        
    def build_target(self, target_name: str, items: list):
        """
            Creates the classification target (e.g. concatenation of labels in data dict)
        """
        n = len(self.data[items[0]])
        self.data[target_name] = ['/'.join([str(self.data[t][i]) for t in items]) for i in range(n)]
           
            
    def collate_fn(self, batch):
        """
            Default collate_fn for classification
        """
        batch_dict = {}

        batch_dict['data'] = torch.cat([self.transforms(b[0]).unsqueeze(0) for b in batch])
        if self.target is not None:
            batch_dict['target'] = torch.tensor([self.voc['target'].w2i(b[1]) for b in batch])
        return batch_dict
        
        
    def load(self):
        """
            Load all the dataset into memory beforehand
        """
        print('preload option is True. Loading images to memory:')
        self.data['im'] = []
        
        for i in tqdm(range(self.length)):
            im = self.process_im(i)
            self.data['im'].append(im)
          
        
    def process_im(self, i):
        """
            Open, crop (if bbox available)
        """
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        if 'bbox' in self.data.keys():
            bbox = self.data['bbox'][i]
            im   = im.crop(bbox)  
        return im
            
        
    def __getitem__(self, i):
        if self.preload:
            if self.target is None:
                return (self.data['im'][i],)
            else:
                return self.data['im'][i], self.data['target'][i]
        else:
            if self.target is None:
                return (self.process_im(i),)
            else:
                return self.process_im(i), self.data['target'][i]               
    
    
    def __len__(self):
        return self.length
    

    def verbose(self):
        s = "Dataset with {} items.\n".format(self.length)
        s += "Metadata available per item: "
        s += ', '.join(self.data.keys())
        s += '\n'
        print(s)
