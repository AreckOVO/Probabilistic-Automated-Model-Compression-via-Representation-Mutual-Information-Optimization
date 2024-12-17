#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet20" \
 --path "Exp_base/resnet20_base" \
 --dataset "cifar10" \
 --save_path '/home/data/cifar10' \
 --target_bops 500000000 \
 --beta 246 \
 --add_channel_rate 0.1 \
 --quant_a_bits 8 \