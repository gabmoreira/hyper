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

from voc import Vocabulary
from PIL import ImageEnhance

transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")
        return out
    
    
def get_cub_transforms(split: str = None, size=84):
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
    
    
    
def get_df_transforms(split: str, size=224):
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



class CUBData(Dataset):
    def __init__(self,
                 img_path: str,
                 data_dict_path: str,
                 im_padding: bool,
                 transforms):
 
        self.img_path   = img_path
        self.transforms = transforms
        self.im_padding = im_padding
        
        """
            Reads dict with all data from data_dict_path

            Examples:
                dict['path']       = ['images/159.Black_and_white_Warbler/Black_...jpg', ...]    
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
        self.verbose()
        
        
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
        
        batch_dict['data']   = torch.cat([self.transforms(b[0]).unsqueeze(0) for b in batch])
        batch_dict['target'] = torch.tensor([self.voc['target'].w2i(b[1]) for b in batch])

        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """ 
        im = self.process_im(i)
        target = self.data['target'][i]
        
        return im, target
    
    
    def __len__(self):
        return self.length
    
    
    def pad_im(self, im):
        """
            Resizes PIL image as square image with padding
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
    
    
    def process_im(self, i, min_bbox_width=100, min_bbox_height=100):
        """
            Open, crop, resize i-th image
        """
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        bbox = self.data['bbox'][i]
        im   = im.crop(bbox)
        
        if self.im_padding:
            im = self.pad_im(im)    
    
        return im
    

    def verbose(self):
        """
            To make sure everything is ok
        """
        s = "Dataset with {} datapoints \nLabels: ".format(self.length)
        s += '\n\n'
        s += "Metadata available per datapoint: "
        s += ', '.join(self.data.keys())
        print(s)
    
            
    
class DeepFashionData(Dataset):
    def __init__(self,
                 img_path: str,
                 data_dict_path: str,
                 im_padding: bool,
                 transforms):
 
        self.img_path   = img_path
        self.transforms = transforms
        self.im_padding = im_padding
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
        
        batch_dict['data']   = torch.cat([self.transforms(b[0]).unsqueeze(0) for b in batch])
        batch_dict['target'] = torch.tensor([self.voc['target'].w2i(b[1]) for b in batch])

        return batch_dict
        
        
    def __getitem__(self, i):
        """
        """ 
        im = self.process_im(i)
        target = self.data['target'][i]
        
        return im, target
    
    
    def __len__(self):
        return self.length
    
    
    def pad_im(self, im):
        """
            Resizes PIL image as square image with padding
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
    
    
    def process_im(self, i, min_bbox_width=100, min_bbox_height=100):
        """
            Open, crop, resize i-th image
        """
        im = Image.open(os.path.join(self.img_path, self.data['path'][i]))
        im = im.convert(mode='RGB')
        
        bbox = self.data['bbox'][i]
        im   = im.crop(bbox)
        
        if self.im_padding:
            im = self.pad_im(im)    
    
        return im
    