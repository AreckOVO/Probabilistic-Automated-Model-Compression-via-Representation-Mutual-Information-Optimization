#!/usr/bin/env bash
# code_path='/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "mobilenetv2" \
 --path "Exp_base/mobilenetv2_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --test_batch_size 60 \
 --target_bops 10000000000 \
 --beta 243 \
 --add_channel_rate 0.15 \
 --quant_a_bits 8 \