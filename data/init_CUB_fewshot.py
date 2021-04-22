import os
import torch
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import shutil
import yaml
sys.path.append('..')
from utils import util

with open('../config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.abspath(config['data_path'])
origin_path = os.path.join(data_path,'CUB_200_2011')
raw_path = os.path.join(data_path,'CUB_fewshot_raw')
cropped_path = os.path.join(data_path,'CUB_fewshot_cropped')

util.mkdir(raw_path)
util.mkdir(cropped_path)

id2path = {}

with open(os.path.join(origin_path,'images.txt')) as f:
    lines = f.readlines()
    for line in lines:
        index, path = line.strip().split()
        index = int(index)
        id2path[index] = path

cat2name = {}

with open(os.path.join(origin_path,'classes.txt')) as f:
    lines = f.readlines()
    for line in lines:
        cat, name = line.strip().split()
        cat = int(cat)
        cat2name[cat] = name

cat2img = {}

with open(os.path.join(origin_path,'image_class_labels.txt')) as f:
    lines = f.readlines()
    for line in lines:
        image_id, class_id = line.strip().split()
        image_id = int(image_id)
        class_id = int(class_id)
        
        if class_id not in cat2img:
            cat2img[class_id]=[]
        cat2img[class_id].append(image_id)


id2bbx={}

with open(os.path.join(origin_path,'bounding_boxes.txt')) as f:
    lines = f.readlines()
    for line in lines:
        index,x,y,width,height = line.strip().split()
        index = int(index)
        x = float(x)
        y = float(y)
        width = float(width)
        height = float(height)
        id2bbx[index] = [x,y,width,height]


train_cat = []
val_cat = []
test_cat = []

train = []
val = []
test = []

for i in range(1,201):
    img_list = cat2img[i]
    name = cat2name[i]
    if i%2 == 0:
        train_cat.append(name)
        train.extend(img_list)
    elif i%4 == 1:
        val_cat.append(name)
        val.extend(img_list)
    elif i%4 ==3:
        test_cat.append(name)
        test.extend(img_list)


split = ['train','val','test']
split_cat = [train_cat,val_cat,test_cat]
split_img = [train,val,test]


print('organizing CUB_fewshot_raw ...')
for i in range(3):

    split_path = os.path.join(raw_path,split[i])
    util.mkdir(split_path)
    
    for cat_name in split_cat[i]:
        util.mkdir(os.path.join(split_path,cat_name))

    for index in tqdm(split_img[i]):
        img_path = id2path[index]
        origin_img = os.path.join(origin_path,'images',img_path)
        target_img = os.path.join(split_path,img_path)
        shutil.copy(origin_img,target_img)

print('getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=raw_path,transform_type=0)


print('organizing CUB_fewshot_cropped ...')
for i in range(3):

    split_path = os.path.join(cropped_path,split[i])
    util.mkdir(split_path)
    
    for cat_name in split_cat[i]:
        util.mkdir(os.path.join(split_path,cat_name))

    for index in tqdm(split_img[i]):
        img_path = id2path[index]
        origin_img = os.path.join(origin_path,'images',img_path)
        target_img = os.path.join(split_path,img_path[:-3]+'png')

        p = Image.open(origin_img)
        x,y,width,height = id2bbx[index]
        p = p.crop((x,y,x+width,y+height))
        p.save(target_img)

print('getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=cropped_path,transform_type=1)