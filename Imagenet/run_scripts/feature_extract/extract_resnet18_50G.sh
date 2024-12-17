# #!/usr/bin/env bash
# code_path='/home/wmk/codes/new/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
# 原来test_batch_size是250
#用四卡会出错
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet18" \
 --path "Exp_base/resnet18_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --test_batch_size 60 \
 --target_bops 50000000000 \
 --beta 2000 \
 --add_channel_rate 0.1 \
 --quant_a_bits 8 \