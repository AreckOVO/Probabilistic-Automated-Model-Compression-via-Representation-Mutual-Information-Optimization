import os
import json
import argparse

import torch
import torch.nn as nn
from run_manager import RunManager
from nets import ResNet_CIFAR, TrainRunConfig
import pdb

parser = argparse.ArgumentParser()

""" basic config """
parser.add_argument('--path', type=str, default='Exp/debug') #训练好的模型存放的位置，"Exp_train/train_mobilenet_150m_${RANDOM}"
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--manual_seed', default=13, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=100)

""" optimizer config """
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--warm_epoch', type=int, default=5)
parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str,
                    default='bn#bias', choices=[None, 'bn', 'bn#bias'])

""" dataset config """
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'imagenet', 'imagenet10', 'imagenet100'])
parser.add_argument('--save_path', type=str, default='/home/data/cifar10') #放数据集的地方
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=250)

""" model config """
parser.add_argument('--model', type=str, default="resnet20",
                    choices=['resnet20', 'resnet56'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--quant_cfg', type=str, default="None")
parser.add_argument('--base_path', type=str, default=None) #base模型的存放位置,"Exp_base/mobilenet_base"
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--model_init', type=str,
                    default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')

""" runtime config """
parser.add_argument('--n_worker', type=int, default=24)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument("--local_rank", default=0, type=int)

""" quantization config """
# 初始的激活值bit
parser.add_argument('--a_bits', type=int, default=8)
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


if __name__ == '__main__':
    args = parser.parse_args()

#     random.seed(args.manual_seed)
#     torch.manual_seed(args.manual_seed)
#     torch.cuda.manual_seed(args.manual_seed)
#     torch.cuda.manual_seed_all(args.manual_seed)
#     np.random.seed(args.manual_seed)
    torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = True
    os.makedirs(args.path, exist_ok=True)

    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path
    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = TrainRunConfig( #根据args生成run config
        **args.__dict__
    )

    if args.local_rank == 0: #打印run config
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    weight_path = None
    net_origin = None
    
    if args.model == 'resnet20':
        assert args.dataset=='cifar10', 'resnet20 only supports cifar10 dataset'
        net = ResNet_CIFAR(
            depth=20, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
        if args.base_path!=None:
            weight_path = args.base_path+'/checkpoint/model_best.pth.tar'
            net_origin = nn.DataParallel(ResNet_CIFAR(
                depth=20, num_classes=run_config.data_provider.n_classes))
    elif args.model == 'resnet56':
        assert args.dataset=='cifar10', 'resnet56 only supports cifar10 dataset'
        net = ResNet_CIFAR(
            depth=56, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
        if args.base_path!=None:
            weight_path = args.base_path+'/checkpoint/model_best.pth.tar'
            net_origin = nn.DataParallel(ResNet_CIFAR(
                depth=56, num_classes=run_config.data_provider.n_classes))

    # build run manager
    run_manager = RunManager(args.path, net, run_config) #run_manager输出存放路径是args.path，对应的网络是net
    if args.local_rank == 0:
        run_manager.save_config(print_info=True)

    # load checkpoints
    if args.base_path is not None:
        weight_path = args.base_path+'/checkpoint/model_best.pth.tar'

    if args.resume: #True or False，加载resume模型，包括权重、epoch、best_acc和optimizer
        run_manager.load_model() #从args.save_path中加载权重
        if args.train and run_manager.best_acc == 0: #还没开始训练且需要训练
            loss, acc1, acc5 = run_manager.validate( #先validate一次
                is_test=True, return_top5=True)
            run_manager.best_acc = acc1
    elif weight_path is not None and os.path.isfile(weight_path): #从weight_path加载预训练模型
        assert net_origin is not None, "original network is None"
        net_origin.load_state_dict(torch.load(weight_path)['state_dict'])
        net_origin = net_origin.module #从dataparellel中获得模型
        if args.model == 'resnet20':
            run_manager.reset_model(ResNet_CIFAR(
                num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg), depth=20, cutout=True), model_origin = net_origin.cpu(), quant_cfg=eval(args.quant_cfg), args=args)
        elif args.model == 'resnet56':
            run_manager.reset_model(ResNet_CIFAR(
                num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg), depth=56, cutout=True), model_origin = net_origin.cpu(), quant_cfg=eval(args.quant_cfg), args=args)
    else:
        print('Random initialization')

    print('model after compression...\n')
    # train
    if args.train:
        print('Start training: %d' % args.local_rank)
        if args.local_rank == 0:
            print(net)
        run_manager.train(print_top5=True)
        if args.local_rank == 0:
            run_manager.save_model()

    output_dict = {}

    # test
    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)
