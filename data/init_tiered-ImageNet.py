import numpy as np
import pickle as pkl
import cv2
import os
import argparse
from tqdm import tqdm
import yaml
import sys
sys.path.append('..')
from utils import util

# modified based on https://github.com/mileyan/simple_shot/blob/master/src/utils/tieredImagenet.py


def save_imgs(origin_path, target_path, tag):
    data_file = os.path.join(origin_path, tag + '_images_png.pkl')
    label_file = os.path.join(origin_path, tag + '_labels.pkl')
    with open(data_file, 'rb') as f:
        array = pkl.load(f)
    with open(label_file, 'rb') as f:
        labels = pkl.load(f)
    for idx, (img, glabel, slabel) in enumerate(tqdm(zip(array, labels['label_general'], labels['label_specific']), total=len(array))):

        cat_name = str(glabel)+'_'+str(slabel)
        if not os.path.isdir(os.path.join(target_path,tag,cat_name)):
            util.mkdir(os.path.join(target_path,tag,cat_name))

        file_name = os.path.join(target_path,tag,cat_name,str(idx)+'.png')

        im = cv2.imdecode(img, cv2.IMREAD_COLOR)
        cv2.imwrite(file_name, im)


if __name__ == '__main__':

    with open('../config.yml', 'r') as f:
        config = yaml.safe_load(f)
    data_path = os.path.abspath(config['data_path'])

    origin_path = os.path.join(data_path,'tiered-imagenet')
    target_path = os.path.join(data_path,'tiered-ImageNet')

    util.mkdir(target_path)
    util.mkdir(os.path.join(target_path,'train'))
    util.mkdir(os.path.join(target_path,'val'))
    util.mkdir(os.path.join(target_path,'test'))

    save_imgs(origin_path, target_path, 'train')
    save_imgs(origin_path, target_path, 'val')
    save_imgs(origin_path, target_path, 'test')