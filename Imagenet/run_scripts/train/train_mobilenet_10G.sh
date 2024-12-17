#!/usr/bin/env bash
# code_path='/home/wmk/codes/new/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenet" \
 --cfg "[22, 64, 64, 64, 180, 128, 512, 256, 256, 335, 256, 435, 512, 870]" \
 --quant_cfg "[4, 4, 5, 5, 4, 8, 7, 7, 6, 4, 4, 4, 4, 7]" \
 --path "Exp_train/train_mobilenet_50m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --base_path "Exp_base/mobilenet_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 250 \
 --train_batch_size 256 \
 --label_smoothing 0.1 \
 --quant_a_bits 8 \
 --q_type 0 \
 --q_level 0 \
 --weight_observer 0 \
