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
  --cfg 16,16,16,11,11,12,12,26,26,31,31,24,24,50,50,38,38,32,32 \
  --quant_cfg 8,5,5,7,7,7,7,4,4,4,4,5,5,4,4,4,4,4,4 \
  --quant_a_bits 8 \
  --q_type 0 \
  --q_level 0 \
  --weight_observer 0 \
