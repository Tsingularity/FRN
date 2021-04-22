#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 3 \
    --val_epoch 20 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 30 \
    --train_shot 1 \
    --train_transform_type 0 \
    --test_shot 1 \
    --pre \
    --gpu 0
