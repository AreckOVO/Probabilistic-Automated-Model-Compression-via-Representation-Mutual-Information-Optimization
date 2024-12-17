#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 random_compress.py \
 --model "resnet20" \
 --path "Exp_base/resnet20_base" \
 --dataset "cifar10" \
 --save_path '/home/data/cifar10' \
 --target_bops 2000000000 \
 --beta 246 \
 --add_channel 6 \
 --quant_a_bits 8 \
