# !/usr/bin/env bash
# code_path='/home/wmk/codes/new/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "resnet18" \
 --cfg "[32, 44, 44, 54, 108, 65, 70, 236, 130, 256, 502, 424, 482]" \
 --quant_cfg "[6, 7, 4, 4, 4, 4, 8, 5, 7, 5, 4, 7, 4]" \
 --path "Exp_train/train_resnet18_50G_${RANDOM}" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet/' \
 --sync_bn \
 --warm_epoch 1 \
 --base_path "Exp_base/resnet18_base" \
 --n_epochs 120 \
 --train_batch_size 256 \
 --quant_a_bits 8 \
 --q_type 0 \
 --q_level 0 \
 --weight_observer 0 \
