import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
from models.FRN import FRN


args = trainer.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'tiered-ImageNet')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_loader = dataloaders.normal_train_dataloader(data_path=pm.train,
                                                batch_size=args.batch_size,
                                                transform_type=args.train_transform_type)

num_cat = len(train_loader.dataset.classes)

model = FRN(is_pretraining=True,
            num_cat=num_cat,
            resnet=args.resnet)

train_func = partial(frn_train.pre_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)