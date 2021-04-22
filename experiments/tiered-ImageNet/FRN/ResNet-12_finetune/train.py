import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
from models.FRN import FRN
from utils import util


args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'tiered-ImageNet')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = FRN(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)

#pretrained_model_path = '../ResNet-12_pretrain/model_ResNet-12.pth'
pretrained_model_path = '../../../../trained_model_weights/tiered-ImageNet/FRN/ResNet-12_pretrain/model.pth'

model.load_state_dict(torch.load(pretrained_model_path,map_location=util.get_device_map(args.gpu)),strict=False)

train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)