"""
    transforms.py
    Apr 24 2023
    Gabriel Moreira
"""
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)


def imshow(tensor: torch.Tensor, i: int=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    if tensor.ndim == 3:
        if tensor.shape[0] == 3:
            im = tensor.permute(1,2,0)
            ax.imshow((im-im.min())/(im.max()-im.min()))
        elif tensor.shape[-1] == 3:
            ax.imshow((tensor-tensor.min())/(tensor.max()-tensor.min()))
    else:
        if tensor.shape[1] == 3:
            im = tensor[i,...].permute(1,2,0)
            ax.imshow((im-im.min())/(im.max()-im.min()))
        elif tensor.shape[-1] == 3:
            ax.imshow((tensor[i,...]-tensor[i,...].min())/(tensor[i,...].max()-tensor[i,...].min()))    



class SquarePadding(object):
    """
        Pads image to make it square
        Keeps aspect ratio
        Returns a PIL image
    """
    def __init__(self, fill=0):
        self.fill = fill
    
    def __call__(self, im):
        old_size = torch.tensor(im.size[::-1])
        d = torch.argmax(old_size)
        
        new_size = old_size.max()

        pad          = int(new_size - old_size[1-d])
        padding      = [0,0,0,0]
        padding[d]   = pad // 2
        padding[d+2] = new_size - old_size[1-d] - padding[d]

        new_im = fn.pad(im, padding, fill=self.fill) 
        
        return new_im
    

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
    
    
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
    t_train = T.Compose([SquarePadding(),
                         T.RandomResizedCrop(size),
                         ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                         T.RandomHorizontalFlip(),
                         T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    t_val = T.Compose([SquarePadding(),
                       T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(size),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if split == 'train':
        return t_train
    elif split == 'test' or split == 'val':
        return t_val
    else:
        return t_train, t_val
