#!/bin/bash
python train.py \
     --opt sgd \
     --lr 1e-1 \
     --gamma 1e-1 \
     --epoch 30 \
     --stage 3 \
     --batch_size 128 \
     --val_epoch 5 \
     --weight_decay 5e-4 \
     --nesterov \
     --train_transform_type 0 \
     --resnet \
     --train_shot 1 \
     --test_shot 1 5 \
     --pre \
     --gpu 0