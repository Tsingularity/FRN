import os
import json
import torch
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import shutil
import yaml
sys.path.append('..')
from utils import util
# import util

with open('../config.yml','r') as f:
# with open('config.yml','r') as f:
    config = yaml.safe_load(f)
    
data_path = os.path.abspath(config['data_path'])
origin_path = os.path.join(data_path,'inat2017')
imgfolder = 'inat2017_84x84'
img_path = os.path.join(data_path,imgfolder)
rel_path = os.path.join('..','..','..',imgfolder)
inat_path = os.path.join(data_path,'meta_iNat')
tier_path = os.path.join(data_path,'tiered_meta_iNat')

util.mkdir(img_path)
util.mkdir(inat_path)
util.mkdir(tier_path)

with open(os.path.join(origin_path,'train_2017_bboxes.json')) as f:
    allinfo = json.load(f)
annolist = allinfo['annotations']

annodict = dict() # im_id to list of box_ids
boxdict = dict() # box_id to box coords
catdict = dict() # dict of numerical category codes / labels to corresponding list of image ids
for d in annolist:
    im = d['image_id']
    boxid = d['id']
    cat = d['category_id']
    
    # Add box_id to image entry
    if im in annodict:
        annodict[im].append(boxid)
    else:
        annodict[im] = [boxid]
        
    # Add mapping from box_id to box
    boxdict[boxid] = d['bbox']
    
    # Add image to category set
    if cat in catdict:
        catdict[cat].add(im)
    else:
        catdict[cat] = set([im])
        
        
# assemble im_id -> filepath dictionary

namelist = allinfo['images']
keys = []
vals = []
for d in namelist:
    keys.append(d['id'])
    vals.append(os.path.join(origin_path,d['file_name']))
pather = dict(zip(keys,vals))


# Pare down the category dictionary to the desired size

clist = list(catdict.keys())
for c in clist:
    if len(catdict[c]) < 50 or len(catdict[c]) > 1000:
        catdict.pop(c)

supercat = dict()
for d in allinfo['categories']:
    catid = d['id']
    if catid in catdict:
        sc = d['supercategory']
        if sc in supercat:
            supercat[sc].append(catid)
        else:
            supercat[sc] = [catid,]

    
# shrink images
catlist = list(catdict.keys())
boxdict_shrunk = dict() # abbreviated path -> [box corners]
pather_shrunk = dict() # im_id -> new path (relative, for symlinks)
print('Shrinking images to 84x84 ...')
for c in tqdm(catlist):
    # For each category:
    catpath = os.path.join(img_path,str(c))
    if not os.path.exists(catpath):
        os.makedirs(catpath)
    ims = catdict[c]
    for imkey in ims:
        # For each image:
        path = pather[imkey]
        file = path[path.rfind(os.path.sep)+1:path.rfind('.')]+'.png'
        fname = os.path.join(str(c),file)
        newpath = os.path.join(catpath,fname)
        pather_shrunk[imkey] = os.path.join(rel_path,fname)
        # Downsize the image to 84x84
        with open(path, 'rb') as f:
            p = Image.open(f)
            w,h = p.size
            p = p.convert('RGB')
        p = p.resize((84,84), Image.BILINEAR)
        p.save(newpath)
        # Downsize the bounding box annotations to 10x10
        boxes = annodict[imkey]
        boxdict_shrunk[str(c)+'/'+file] = []
        for boxcode in boxes:
            box = boxdict[boxcode]
            xmin = box[0]
            xmax = box[2]+xmin
            ymin = box[1]
            ymax = box[3]+ymin
            boxdict_shrunk[str(c)+'/'+file].append([xmin*10/w, ymin*10/h, xmax*10/w, ymax*10/h])
torch.save(boxdict_shrunk, os.path.join(img_path,'box_coords.pth'))


def makedataset(traincatlist, testcatlist, datapath, catdict, pather):
    
    def makesplit(catlist, datapath, split, catdict, pather, imsplit):
        splitpath = os.path.join(datapath,split)
        util.mkdir(splitpath)
        for c in catlist:
            # For each category:
            catpath = os.path.join(splitpath,str(c))
            if not os.path.exists(catpath):
                os.makedirs(catpath)
            ims = list(catdict[c])
            ims = imsplit(ims)
            for imkey in ims:
                path = pather[imkey]
                newpath = os.path.join(catpath,path[path.rfind(os.path.sep)+1:path.rfind('.')]+'.png')
                os.symlink(path, newpath)
    
    makesplit(traincatlist, datapath, 'train', catdict, pather, lambda x: x)
    makesplit(testcatlist, datapath, 'test', catdict, pather, lambda x: x)
    makesplit(testcatlist, datapath, 'refr', catdict, pather, lambda x: x[:len(x)//5])
    makesplit(testcatlist, datapath, 'query', catdict, pather, lambda x: x[len(x)//5:])

# meta-iNat
print('Organizing meta-iNat ...')
split_folder = os.path.abspath('./meta_iNat_split/')
traincatlist = torch.load(os.path.join(split_folder,'meta_iNat_traincats.pth'))
testcatlist = torch.load(os.path.join(split_folder,'meta_iNat_testcats.pth'))
makedataset(traincatlist, testcatlist, inat_path, catdict, pather_shrunk)
torch.save(boxdict_shrunk, os.path.join(inat_path,'box_coords.pth'))

# tiered meta-iNat
print('Organizing tiered meta-iNat ...')
traincatlist = (supercat['Animalia']+supercat['Aves']+supercat['Reptilia']+supercat['Amphibia']
                +supercat['Mammalia']+supercat['Actinopterygii']+supercat['Mollusca'])
testcatlist = supercat['Insecta']+supercat['Arachnida']
makedataset(traincatlist, testcatlist, tier_path, catdict, pather_shrunk)
torch.save(boxdict_shrunk, os.path.join(tier_path,'box_coords.pth'))

print('Organizing complete!')

