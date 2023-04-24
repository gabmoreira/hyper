"""
    transforms.py
    Mar 4 2023
    Gabriel Moreira
"""
import numpy as np

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as fn
import torchvision.transforms as T

transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)


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

class VICTransform(object):
    def __init__(self, size: int):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4,
                                             contrast=0.4,
                                             saturation=0.2,
                                             hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_prime = T.Compose(
            [
                T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4,
                                             contrast=0.4,
                                             saturation=0.2,
                                             hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
    
    
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


