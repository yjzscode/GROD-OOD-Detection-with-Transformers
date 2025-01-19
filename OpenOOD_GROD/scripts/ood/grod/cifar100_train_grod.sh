#!/bin/bash
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ood/grod/cifar100_train_grod.sh &
GPU=1
CPU=1
node=73
jobname=openood


for warmup_threshold in 40
do
    for stat_smooth in 0.3
    do
        python main.py \
            --config configs/datasets/cifar100/cifar100.yml \
            configs/networks/grod_net.yml \
            configs/pipelines/train/train_grod.yml \
            configs/preprocessors/base_preprocessor.yml \
            --network.backbone.name vit-b-16 \
            --network.pretrained False \
            --dataset.train.batch_size 64 \
            --optimizer.num_epochs 12 \
            --optimizer.lr 0.0001 \
            --optimizer.weight_decay 0.05 \
            --trainer.gamma 0.1 \
            --trainer.alpha 0.001 \
            --trainer.stat_smooth $stat_smooth \
            --trainer.warmup_threshold $warmup_threshold \
            --seed 42
    done
done



    



