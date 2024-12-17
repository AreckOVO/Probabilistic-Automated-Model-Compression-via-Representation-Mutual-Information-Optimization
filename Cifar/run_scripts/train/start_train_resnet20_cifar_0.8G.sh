## ============================= resnet cifar10 decode  ============================
#gpu用四卡跑会出错
python main.py \
  --gpu_id 3 \
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
  --cfg 16,16,16,16,16,10,10,26,26,20,20,21,21,64,64,44,44,34,34 \
  --quant_cfg 8,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 \
  --quant_a_bits 8 \
  --q_type 0 \
  --q_level 0 \
  --weight_observer 0 \

