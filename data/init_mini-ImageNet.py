from PIL import Image
import torch
import os
import numpy as np
import sys
from tqdm import tqdm
import argparse
import shutil
import yaml

sys.path.append('..')
from utils import util

with open('../config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.abspath(config['data_path'])

split = ['train','val','test']
image_folder = os.path.join(data_path,'images/')
csv_folder = os.path.abspath('./mini-ImageNet_split/')
target_folder = os.path.join(data_path,'mini-ImageNet')
util.mkdir(target_folder)

cat_list = []

for i in split:
    
    util.mkdir(os.path.join(target_folder,i))

    csv_file = os.path.join(csv_folder,i+'.csv')
    
    with open(csv_file,'r') as f:
        for line in tqdm(f.readlines()[1:]):
            
            line = line.strip().split(',')
            img_path = line[0]
            cat = line[1]
            
            if cat not in cat_list:
                cat_list.append(cat)
                util.mkdir(os.path.join(target_folder,i,cat))

            shutil.copy(os.path.join(image_folder,img_path),os.path.join(target_folder,i,cat,img_path))


print('getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=target_folder,transform_type=0)
