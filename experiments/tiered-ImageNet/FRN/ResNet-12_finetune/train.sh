#!/bin/bash
python train.py \
     --opt sgd \
     --lr 1e-3 \
     --gamma 1e-1 \
     --epoch 60 \
     --decay_epoch 30 50 \
     --val_epoch 5 \
     --weight_decay 5e-4 \
     --nesterov \
     --train_transform_type 1 \
     --test_transform_type 2 \
     --resnet \
     --train_shot 5 \
     --train_way 20 \
     --test_shot 1 5 \
     --gpu 0
