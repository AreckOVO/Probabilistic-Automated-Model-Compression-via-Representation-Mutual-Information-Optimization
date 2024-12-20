# #!/usr/bin/env bash
# code_path='/home/wmk/codes/new/ITPruner/Imagenet/'
# chmod +x ${code_path}/prep_imagenet.sh
# cd ${code_path}
# echo "preparing data"
# bash ${code_path}/prep_imagenet.sh >> /dev/null
# echo "preparing data finished"
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet50" \
 --path "Exp_base/resnet50_base" \
 --dataset "imagenet" \
 --save_path '/home/data/imagenet' \
 --test_batch_size 4 \
 --target_flops 1800000000 \
 --beta 2000