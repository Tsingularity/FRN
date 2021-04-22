import os
import torch
import math
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import shutil
import yaml
sys.path.append('..')
from utils import util

np.random.seed(42)

with open('../config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.abspath(config['data_path'])

home_dir = os.path.join(data_path,'fgvc-aircraft-2013b/data')
target_dir = os.path.join(data_path,'Aircraft_fewshot')

util.mkdir(target_dir)

cat_id2name={}
cat_name2id={}
with open(os.path.join(home_dir,'variants.txt')) as f:
    content = f.readlines()
    for i in range(len(content)):
        name=content[i].strip()
        cat_id2name[i]=name
        cat_name2id[name]=i
        
img2cat={}
cat2img={}
for i in ['images_variant_trainval.txt','images_variant_test.txt']:
    with open(os.path.join(home_dir,i)) as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            img = line[:7]
            cat = line[8:]
            cat_id = cat_name2id[cat]
            img2cat[img] = cat_id
            if cat_id not in cat2img:
                cat2img[cat_id]=[]
            cat2img[cat_id].append(img)

img2bbx={}
with open(os.path.join(home_dir,'images_box.txt')) as f:
    content = f.readlines()
    for line in content:
        line = line.strip()
        img,xmin,ymin,xmax,ymax = line.split()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        img2bbx[img] = [xmin,xmax,ymin,ymax]

train_cat=[]
val_cat=[]
test_cat=[]
for i in range(100):
    if i%2==0:
        train_cat.append(i)
    elif i%4==1:
        val_cat.append(i)
    elif i%4==3:
        test_cat.append(i)

dir_name = ['train','val','test']

for i in dir_name:
    util.mkdir(os.path.join(target_dir,i))

cat_list = [train_cat,val_cat,test_cat]

for i in range(3):
    for j in cat_list[i]:
        util.mkdir(os.path.join(target_dir,dir_name[i],str(j)))

for i in range(3):
    for j in tqdm(cat_list[i]):
        for img in cat2img[j]:
            img_path = os.path.join(home_dir,'images',img+'.jpg')
            target_path = os.path.join(target_dir,dir_name[i],str(j),img+'.png')
            
            image = Image.open(img_path)

            bbx = img2bbx[img]

            image = image.crop((bbx[0],bbx[2],bbx[1],bbx[3]))
            image.save(target_path)

print('getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=target_dir,transform_type=1)