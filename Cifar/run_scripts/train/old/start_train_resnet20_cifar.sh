## ============================= resnet cifar10 decode  ============================
#gpu用四卡跑会出错
python main.py \
  --gpu_id 1 \
  --exec_mode train \
  --learner vanilla \
  --dataset cifar10 \
  --data_path /home/data/cifar10 \
  --model_type resnet_decode \
  --lr 0.1 \
  --lr_min 0. \
  --lr_decy_type cosine \
  --weight_decay 5e-4 \
  --nesterov \
  --epochs 300 \
  --cfg 16,11,11,11,11,11,11,22,22,21,21,22,22,45,45,43,43,48,48 \
  # --quant_cfg 8,8,8,4,4,8,8,8,8,5,5,8,8,8,8,8,8,8,8

