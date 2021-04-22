import os
import math
import torch
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
from . import samplers,transform_manager


def get_dataset(data_path,is_training,transform_type,pre):

    dataset = datasets.ImageFolder(
        data_path,
        loader = lambda x: image_loader(path=x,is_training=is_training,transform_type=transform_type,pre=pre))

    return dataset



def meta_train_dataloader(data_path,way,shots,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = samplers.meta_batchsampler(data_source=dataset,way=way,shots=shots),
        num_workers = 3,
        pin_memory = False)

    return loader



def meta_test_dataloader(data_path,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

    dataset = get_dataset(data_path=data_path,is_training=False,transform_type=transform_type,pre=pre)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = samplers.random_sampler(data_source=dataset,way=way,shot=shot,query_shot=query_shot,trial=trial),
        num_workers = 3,
        pin_memory = False)

    return loader


def normal_train_dataloader(data_path,batch_size,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 3,
        pin_memory = False,
        drop_last=True)

    return loader


def image_loader(path,is_training,transform_type,pre):

    p = Image.open(path)
    p = p.convert('RGB')

    final_transform = transform_manager.get_transform(is_training=is_training,transform_type=transform_type,pre=pre)

    p = final_transform(p)

    return p
