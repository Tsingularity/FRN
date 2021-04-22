from PIL import Image
import torch
import os
import numpy as np
import sys
import argparse
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms

def mkdir(path):
    
    if os.path.exists(path): 
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder,transform_type):
    split = ['val','test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(92),
                                    transforms.CenterCrop(84)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize([92,92]),
                                    transforms.CenterCrop(84)])

    cat_list = []

    for i in split:
        
        cls_list = os.listdir(os.path.join(image_folder,i))

        folder_name = i+'_pre'

        mkdir(os.path.join(image_folder,folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder,folder_name,j))

            img_list = os.listdir(os.path.join(image_folder,i,j))

            for img_name in img_list:
        
                img = Image.open(os.path.join(image_folder,i,j,img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder,folder_name,j,img_name[:-3]+'png'))


def get_device_map(gpu):
    cuda = lambda x: 'cuda:%d'%x
    temp = {}
    for i in range(4):
        temp[cuda(i)]=cuda(gpu)
    return temp