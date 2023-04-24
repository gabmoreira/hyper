"""
    loader.py
    Mar 4 2023
    Gabriel Moreira
"""
import os
import math
import torch
import numpy as np
from PIL import Image, ImageEnhance

from tqdm.auto import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn
import torchvision.transforms as T

from voc import Vocabulary

from typing import List

transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)

    
class ImageJitter(object):
    def __init__(self, transformdict):
        """
        """
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        """
        """
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")
        return out
    
    
def get_cub_transforms(split: str=None, size: int=84):
    """
    """
    t_train = T.Compose([T.RandomResizedCrop(size),
                         ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                         T.RandomHorizontalFlip(),
                         T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    t_val = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(size),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if split == 'train':
        return t_train
    elif split == 'test' or split == 'val':
        return t_val
    else:
        return t_train, t_val
    
    
    
def get_df_transforms(split: str, size: int=224):
    """
    """
    t_train = T.Compose([T.RandomResizedCrop(size, scale=(0.2, 1.)),
                         T.RandomHorizontalFlip(),
                         T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                         T.RandomGrayscale(p=0.2),
                         T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    t_val = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if split == 'train':
        return t_train
    elif split == 'test' or split == 'val':
        return t_val
    else:
        return t_train, t_val



    
    
class ImSamples(Dataset):
    def __init__(self,
                 img_path: str,
                 transforms,
                 data_dict_path: str=None,
                 im_padding: bool=False,
                 target: List[str]=None,
                 preload: bool=False):
 
        self.img_path   = img_path
        self.transforms = transforms
        self.im_padding = im_padding
        self.preload    = preload
        self.target     = target
        
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
          
        
    def process_im(self, i, min_bbox_width=100, min_bbox_height=100):
        """
            Open, crop (if bbox available), pad the i-th image
        """
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        if 'bbox' in self.data.keys():
            bbox = self.data['bbox'][i]
            im   = im.crop(bbox)
        
        if self.im_padding:
            im = self.pad_im(im)    
    
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
    
    
    def pad_im(self, im):
        """
            Pads image to make it square
            Keeps aspect ratio
            Returns a PIL image
        """
        old_size = torch.tensor(im.size[::-1])
        d = torch.argmax(old_size)
        
        new_size = old_size.max()

        pad          = new_size - old_size[1-d]
        padding      = [0,0,0,0]
        padding[d]   = pad // 2
        padding[d+2] = new_size - old_size[1-d] - padding[d]

        new_im = fn.pad(im, padding, fill=255) 
        
        return new_im
    

    def verbose(self):
        s = "Dataset with {} items.\n".format(self.length)
        s += "Metadata available per item: "
        s += ', '.join(self.data.keys())
        s += '\n'
        print(s)
