"""
    loader.py
    Feb 21 2023
    Gabriel Moreira
"""
import os
import math
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn
import torchvision.transforms as T

from hyper.voc import Vocabulary
    
    
def get_cub_transforms(split: str):
    t_train = T.Compose([T.RandomResizedCrop(84),
                         T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.4),
                         T.RandomHorizontalFlip(),
                         T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    t_val = T.Compose([T.Resize(84, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(84),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

    if split == 'train':
        return t_train
    elif split == 'test' or split == 'val':
        return t_val
    else:
        return t_train, t_val
    
    
    
def get_df_transforms(split: str):
    t_train = T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.)),
                         T.RandomHorizontalFlip(),
                         T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                         T.RandomGrayscale(p=0.2),
                         T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    t_val = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if split == 'train':
        return t_train
    elif split == 'test' or split == 'val':
        return t_val
    else:
        return t_train, t_val



class CUBData(Dataset):
    def __init__(self,
                 img_path: str,
                 data_dict_path: str,
                 im_size: int,
                 transforms):
 
        self.img_path   = img_path
        self.transforms = transforms
        self.im_size    = im_size
        
        """
            Reads dict with all data from data_dict_path

            Examples:
                dict['path']       = ['images/159.Black_and_white_Warbler/Black_And_White_Warbler_0031_160773.jpg', ...]    
                dict['attributes'] = [[1 0 22 3 0 4 4 0 ...], ... ]
                dict['class']      = ['159.Black_and_white_Warbler', ... ]
                dict['bbox']       = [[50, 40, 140, 180], [10, 55, 78, 180], ...]
        """
        data_dict = torch.load(data_dict_path)
        
        self.data = {}
        for key, value in data_dict.items():
            self.data[key] = value
            
        self.build_target('target', ['class'])
        self.build_vocs(['target'])
        
        self.length = len(self.data['target'])
        
        
    def build_vocs(self, items: list):
        """
            Creates a dictionary of vocabularies for the items requested
        """
        self.voc = {item : Vocabulary(self.data[item]) for item in items}

        
    def build_target(self, target_name: str, items: list):
        n = len(self.data[items[0]])
        self.data[target_name] = ['/'.join([str(self.data[t][i]) for t in items]) for i in range(n)]
           
            
    def collate_fn(self, batch):
        """
            Create batch tensors from argument batch (list)
        """
        batch_dict = {}
        
        batch_dict['target'] = torch.tensor([self.voc['target'].w2i(b[0]) for b in batch])
        batch_dict['img']    = torch.cat([self.transforms(b[1]).unsqueeze(0) for b in batch])

        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """ 
        im = self.process_im(i)
        target = self.data['target'][i]
        
        return im, target
    
    
    def __len__(self):
        return self.length
    
    
    def no_distort_resize(self, im, new_size):
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
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        bbox = self.data['bbox'][i]
                
        # Set minimum size for bounding box - some of them are way too small
        bbox_width  = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        margin_x    = max((min_bbox_width-bbox_width) // 2, 0)
        margin_y    = max((min_bbox_height-bbox_height) // 2, 0)
        new_bbox    = [max(bbox[0]-margin_x,0), max(bbox[1]-margin_y,0),
                       min(bbox[2]+margin_x, im.size[0]), min(bbox[3]+margin_y, im.size[1])]

        im = im.crop(new_bbox)
        
        if self.im_size is not None
            im = self.no_distort_resize(im, self.im_size)    
    
        return im
    
    
            
    
class DeepFashionData(Dataset):
    def __init__(self,
                 img_path: str,
                 data_dict_path: str,
                 im_size: int,
                 transforms):
 
        self.img_path   = img_path
        self.transforms = transforms
        
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
        data_dict = torch.load(data_dict_path)
        
        self.data = {}
        for key, value in data_dict.items():
            self.data[key] = value
            
        self.build_target('target', ['gender', 'cat'])
        self.build_vocs(['target'])
        
        self.length = len(self.data['target'])
        
        
    def build_vocs(self, items: list):
        """
            Creates a dictionary of vocabularies for the items requested
        """
        self.voc = {item : Vocabulary(self.data[item]) for item in items}

        
    def build_target(self, target_name: str, items: list):
        n = len(self.data[items[0]])
        self.data[target_name] = ['/'.join([str(self.data[t][i]) for t in items]) for i in range(n)]
           
            
    def collate_fn(self, batch):
        """
            Create batch tensors from argument batch (list)
        """
        batch_dict = {}
        
        batch_dict['target'] = torch.tensor([self.voc['target'].w2i(b[0]) for b in batch])
        batch_dict['img']    = torch.cat([self.transforms(b[1]).unsqueeze(0) for b in batch])

        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """ 
        im = self.process_im(i)
        target = self.data['target'][i]
        
        return im, target
    
    
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
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        bbox = self.data['bbox'][i]
                
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
    