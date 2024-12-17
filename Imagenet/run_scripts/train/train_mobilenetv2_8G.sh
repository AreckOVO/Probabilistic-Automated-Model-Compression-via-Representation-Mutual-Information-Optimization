#!/usr/bin/env bash
# code_path='/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --train \
 --model "mobilenetv2" \
 --cfg "[29, 12, 16, 24, 43, 50, 98, 273]" \
--quant_cfg "[4, 4, 5, 5, 4, 8, 7, 7, 6, 4, 4, 4, 4, 7]" \
 --path "Exp_train/train_mobilenetv2_150m_${RANDOM}" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --base_path "Exp_base/mobilenetv2_base" \
 --warm_epoch 1 \
 --sync_bn \
 --n_epochs 250 \
 --train_batch_size 256 \
 --label_smoothing 0.1 \
 --quant_a_bits 8 \
 --q_type 0 \
 --q_level 0 \
 --weight_observer 0 \
