"""
    deepfashion_loader.py
    Sep 8 2022
    Gabriel Moreira
"""
import os
import math
import torch
import json
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn

from voc import Vocabulary


class DeepFashionData(Dataset):
    def __init__(self,
                 img_path,
                 data_dict_path,
                 taxonomy_path,
                 transforms):
 
        self.img_path   = img_path
        self.transforms = transforms
        self.taxonomy   = None
        
        """
            Load taxonomy dictionary
        """
        if taxonomy_path is not None:
            taxonomy = torch.load(taxonomy_path)
            self.taxonomy = {}
            for key in taxonomy.keys():
                self.taxonomy[key] = torch.tensor(taxonomy[key].astype(np.float32))
            self.taxonomy_depth = max([len(node.split('/')) for node in self.taxonomy.keys()])
            self.taxonomy_dim   = self.taxonomy[list(self.taxonomy.keys())[0]].shape[0]
            
        """
            Reads dictionary with all data from data_dict_path

            Dict-Keys: 
                'path', 'gender', 'cat', 'id', 'filename', 'bbox', 'pose'

            Examples:
                dict['path']     = ['img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg', ...]    
                dict['gender']   = ['WOMEN', 'WOMEN', ... ]
                dict['cat']      = ['Blouses_Shirts', 'Blouses_Shirts', 'Tees_Tanks', ...]
                dict['id']       = ['id_00000001', 'id_00000001', 'id_00000007', ...]
                dict['filename'] = ['02_1_front.jpg', '02_3_back.jpg', ...]
                dict['bbox']     = [[50, 40, 140, 180], [10, 55, 78, 180], ...]
                dict['pose']     = [1, 1, 2, 3, 1, ...]
        """
        data = torch.load(data_dict_path)
        for key, value in data.items():
            setattr(self, key, value)
      
        self.voc = {'gender'    : Vocabulary(self.gender),
                    'cat'       : Vocabulary(self.cat),
                    'body_part' : Vocabulary(self.body_part)}
        
        self.length = len(self.gender)
        
    
    def collate_fn(self, batch):
        """
            Create batch tensors from argument batch (list)
        """
        batch_dict = {}
        
        batch_dict['gender']    = torch.tensor([self.voc['gender'].w2i(b[0]) for b in batch])
        batch_dict['body_part'] = torch.tensor([self.voc['body_part'].w2i(b[1]) for b in batch])
        batch_dict['cat']       = torch.tensor([self.voc['cat'].w2i(b[2]) for b in batch])
        batch_dict['img']       = torch.cat([self.transforms(b[3]).unsqueeze(0) for b in batch])

        if self.taxonomy is not None:
            batch_dict['node_embed'] = torch.cat([b[-1] for b in batch])
            
        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """
        node_embed = None
        if self.taxonomy is not None:
            node_embed = torch.zeros((1, self.taxonomy_dim, self.taxonomy_depth))      
            node_embed[0,:,0] += self.taxonomy[self.gender[i]]
            node_embed[0,:,1] += self.taxonomy[self.gender[i] + '/' + self.body_part[i]]
            node_embed[0,:,2] += self.taxonomy[self.gender[i] + '/' + self.body_part[i] + '/' + self.cat[i]]
            
        im = self.process_im(i)
        
        return self.gender[i], self.body_part[i], self.cat[i], im, node_embed
    
    
    def __len__(self):
        return self.length
    
    
    def no_distort_resize(self, im, new_size=224):
        """
            Resizes PIL image as square image with padding
            Keeps aspect ratio
            Returns a PIL image
        """
        old_size = torch.tensor(im.size[::-1])
        d = torch.argmax(old_size)

        scaling = new_size / old_size[d]

        effective_size      = [0,0]
        effective_size[d]   = new_size
        effective_size[1-d] = int(math.floor(scaling * old_size[1-d]))

        new_im = fn.resize(im, size=effective_size, interpolation=InterpolationMode.BICUBIC)

        pad          = new_size - effective_size[1-d]
        padding      = [0,0,0,0]
        padding[d]   = pad // 2
        padding[d+2] = new_size - effective_size[1-d] - padding[d]

        new_im = fn.pad(new_im, padding, fill=255) 
        
        return new_im
    
    
    def process_im(self, i, min_bbox_width=100, min_bbox_height=100):
        """
            Open, crop, resize i-th image
        """
        im = Image.open(os.path.join(self.img_path, self.path[i]))
        im = im.convert(mode='RGB')
        
        bbox = self.bbox[i]
                
        # Set minimum size for bounding box - some of them are way too small
        bbox_width  = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        margin_x    = max((min_bbox_width-bbox_width) // 2, 0)
        margin_y    = max((min_bbox_height-bbox_height) // 2, 0)
        new_bbox    = [max(bbox[0]-margin_x,0), max(bbox[1]-margin_y,0),
                       min(bbox[2]+margin_x, im.size[0]), min(bbox[3]+margin_y, im.size[1])]

        im = im.crop(new_bbox)
        im = self.no_distort_resize(im)    
    
        return im
    