import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image

def get_transform(is_training=None,transform_type=None,pre=None):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    if transform_type and pre:
        raise Exception('transform_type and pre cannot be specified as True at the same time')

    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,std=std)
                                    ])

    if is_training:

        if transform_type == 0:
            size_transform = transforms.RandomResizedCrop(84)
        elif transform_type == 1:
            size_transform = transforms.RandomCrop(84,padding=8)
        else:
            raise Exception('transform_type must be specified during training!')
        
        train_transform = transforms.Compose([size_transform,
                                            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                            transforms.RandomHorizontalFlip(),
                                            normalize
                                            ])
        return train_transform
    
    elif pre:
        return normalize
    
    else:
        
        if transform_type == 0:
            size_transform = transforms.Compose([transforms.Resize(92),
                                                transforms.CenterCrop(84)])
        elif transform_type == 1:
            size_transform = transforms.Compose([transforms.Resize([92,92]),
                                                transforms.CenterCrop(84)])
        elif transform_type == 2:
            # for tiered-imagenet and (tiered) meta-inat where val/test images are already 84x84
            return normalize

        else:
            raise Exception('transform_type must be specified during inference if not using pre!')
        
        eval_transform = transforms.Compose([size_transform,normalize])
        return eval_transform
