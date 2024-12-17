## ============================= resnet cifar10 decode  ============================
#gpu用四卡跑会出错
python main.py \
  --gpu_id 2 \
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
  --cfg 16,13,13,11,11,14,14,32,32,23,23,21,21,58,58,45,45,51,51 \
  --quant_cfg 8,4,4,7,7,5,5,4,4,7,7,8,8,5,5,4,4,8,8 \
  --quant_a_bits 8 \
  --q_type 0 \
  --q_level 0 \
  --weight_observer 0 \

