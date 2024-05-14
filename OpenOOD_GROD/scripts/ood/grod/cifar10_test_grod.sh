#!/bin/bash
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/ood/grod/cifar10_test_grod.sh & 

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/grod_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 15 \
    --network.pretrained True \
    --network.checkpoint "./results/cifar10_grod_net_num1_grod_al0.001_gm0.1_e20_lr0.0001_default/s42/best.ckpt" \
    --mark 0
