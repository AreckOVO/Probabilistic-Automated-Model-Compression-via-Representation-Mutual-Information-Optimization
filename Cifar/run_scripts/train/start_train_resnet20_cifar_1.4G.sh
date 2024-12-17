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
  --cfg 16,16,16,13,13,15,15,32,32,20,20,26,26,58,58,51,51,64,64 \
  --quant_cfg 8,4,4,7,7,6,6,5,5,8,8,4,4,6,6,7,7,4,4 \
  --quant_a_bits 8 \
  --q_type 0 \
  --q_level 0 \
  --weight_observer 0 \


