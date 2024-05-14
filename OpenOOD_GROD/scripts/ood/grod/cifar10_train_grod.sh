#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ood/grod/cifar10_train_grod.sh &
GPU=1
CPU=1
node=73
jobname=openood

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/grod_net.yml \
    configs/pipelines/train/train_grod.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name vit-b-16 \
    --network.pretrained False \
    --network.feat_dim 1 \
    --dataset.train.batch_size 64 \
    --optimizer.num_epochs 20 \
    --optimizer.lr 0.0001 \
    --optimizer.weight_decay 0.05 \
    --trainer.gamma 0.1 \
    --trainer.alpha 0.001 \
    --seed 42
