import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, proto_train
from datasets import dataloaders
from models.Proto import Proto


args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'meta_iNat')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = Proto(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)

train_func = partial(proto_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)