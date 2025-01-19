#!/bin/bash
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/ood/grod/cifar100_test_grod.sh &


warmup_threshold=40
for stat_smooth in 0.3
do
    checkpoint_path=$(echo './results/cifar100_grod_net_num1_grod_al0.001_gm0.1_stat0.0_thr40_e12_lr0.0001_default/s42/best.ckpt' | sed "s/stat0.0/stat${stat_smooth}/")
    echo $checkpoint_path
    python main.py \
        --config configs/datasets/cifar100/cifar100.yml \
        configs/datasets/cifar100/cifar100_ood.yml \
        configs/networks/grod_net.yml \
        configs/pipelines/test/test_ood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/vim.yml \
        --num_workers 15 \
        --network.pretrained True \
        --trainer.stat_smooth $stat_smooth \
        --trainer.alpha 0.001 \
        --trainer.name grod \
        --trainer.warmup_threshold $warmup_threshold \
        --network.checkpoint $checkpoint_path \
        --mark 0
done




