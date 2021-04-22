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
target_folder = os.path.join(data_path,'tiered-ImageNet_DeepEMD')

print('getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=target_folder,transform_type=0)