#!/bin/bash
python train.py \
    --opt adam \
    --lr 1e-3 \
    --gamma 5e-1 \
    --epoch 20 \
    --stage 5 \
    --val_epoch 2 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 20 \
    --train_shot 5 \
    --train_transform_type 1 \
    --test_transform_type 2 \
    --test_shot 5 \
    --no_val \
    --gpu 0
