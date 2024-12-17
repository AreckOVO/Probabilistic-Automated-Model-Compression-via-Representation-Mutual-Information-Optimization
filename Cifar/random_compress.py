import os
import argparse
import numpy as np
import math
from scipy import optimize
import random, sys
sys.path.append("..")
from CKA import cka
import torch
import torch.nn as nn
from run_manager import RunManager
from nets import ResNet_CIFAR, TrainRunConfig
from utils import *
import copy
import pdb
import random

parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--path', type=str)
parser.add_argument('--model', type=str, default="vgg",
                    choices=['resnet20', 'resnet56'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument("--target_bops", default=0, type=int)
parser.add_argument("--beta", default=1, type=int)

""" dataset config """
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/home/data/cifar10')

""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument("--local_rank", default=0, type=int)

""" quantization config """
# 初始的激活值bit
parser.add_argument('--a_bits', type=int, default=32)
# 初始的激活值bit
parser.add_argument('--quant_a_bits', type=int, default=8)
# 量化方法选择
parser.add_argument('--q_type', type=int, default=0,
                    help='quant_type:0-symmetric, 1-asymmetric')
# 量化级别选择
parser.add_argument('--q_level', type=int, default=0,
                    help='quant_level:0-per_channel, 1-per_layer')
# weight_observer选择
parser.add_argument('--weight_observer', type=int, default=0,
                    help='quant_weight_observer:0-MinMaxObserver, 1-MovingAverageMinMaxObserver')
parser.add_argument('--add_channel', type=int, default=6,
                    help='fine-tune时一次增加几个channel')
if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    # random.seed(args.manual_seed) #不设定随机种子
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path

    run_config = TrainRunConfig(
        **args.__dict__
    )
    if args.local_rank == 0:
        print('Run config:')
        for k, v in args.__dict__.items():
            print('\t%s: %s' % (k, v))

    if args.model == "resnet20":
        assert args.dataset == 'cifar10', 'resnet20 only supports cifar10 dataset'
        net = ResNet_CIFAR(
            depth=20, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "resnet56":
        assert args.dataset == 'cifar10', 'resnet56 only supports cifar10 dataset'
        net = ResNet_CIFAR(
            depth=56, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))

    # build run manager
    run_manager = RunManager(args.path, net, run_config)

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    assert os.path.isfile(best_model_path), 'wrong path'
    if torch.cuda.is_available():
        checkpoint = torch.load(best_model_path, map_location='cpu')
    else:
        checkpoint = torch.load(best_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    run_manager.net.load_state_dict(checkpoint)
    output_dict = {}

    length = len(net.cfg)
    prune_cfg = copy.deepcopy(net.cfg)
    quant_cfg = [8 for i in range(len(net.cfg))]
    sum_bops = net.cfg2bops(prune_cfg, quant_cfg, length, args)    
    print('BOPs',sum_bops / 1e9)


    while(sum_bops > args.target_bops): 
        layer_index = random.randint(0,length-1)   #左闭右闭
        choose_prune = random.choice([True, False])        
        if choose_prune:
            if prune_cfg[layer_index] - args.add_channel > 0:
                prune_cfg[layer_index] -= args.add_channel
        else:
            if quant_cfg[layer_index] > 2:
                quant_cfg[layer_index] -= 1
        sum_bops = net.cfg2bops(prune_cfg, quant_cfg, length, args)  
        print('prune_cfg',prune_cfg)
        print('quant_cfg',quant_cfg)
        print('sum_bops',sum_bops/1e9)
    
    print('last prune_cfg',prune_cfg)
    print('last quant_cfg',quant_cfg)
    print('last sum_bops',sum_bops/1e9)
    # if args.model == "resnet20":
    #     cp_net = ResNet_CIFAR(depth=20, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
    # elif args.model == "resnet56":
    #     cp_net = ResNet_CIFAR(depth=56, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
    # cp_net.init_model(run_config.model_init, run_config.init_div_groups)
    # get_unpruned_weights(cp_net, net) #把模型权重根据L1范数大小迁移到小模型里
    # cp_net.quantize_net(quant_cfg, args)

    # cp_model_name = 'checkpoint.pth.tar'
    # cp_model_path = os.path.join(self.save_path, cp_model_name)
    # torch.save(cp_net, cp_model_path)
        

    

   

            
            

                    


