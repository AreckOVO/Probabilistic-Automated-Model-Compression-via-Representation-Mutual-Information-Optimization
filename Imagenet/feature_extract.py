import os
import argparse
import numpy as np
import math
from scipy import optimize
import random, sys
sys.path.append("../../ITPruner")
from CKA import cka
import torch
import torch.nn as nn
from run_manager import RunManager
from models import ResNet_ImageNet, MobileNet, MobileNetV2, TrainRunConfig
from utils import *
import copy
import pdb

parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--path', type=str)
parser.add_argument('--model', type=str, default="resnet18",
                    choices=['resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobilenet'])
parser.add_argument('--cfg', type=str, default="None")
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument("--target_bops", default=0, type=int)
parser.add_argument("--beta", default=1, type=int)

""" dataset config """
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--save_path', type=str, default='/home/data/imagenet')

""" runtime config """
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--test_batch_size', type=int, default=250)
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
parser.add_argument('--add_channel_rate', type=float, default=0.1,
                    help='fine-tune时一次增加或减少原始channel的多少')

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    # distributed setting
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # prepare run config
    run_config_path = '%s/run.config' % args.path #run_config_path存放lr、epoch等信息    

    run_config = TrainRunConfig(
        **args.__dict__
    )    

    if args.local_rank == 0:
        print('Run config:')
        for k, v in args.__dict__.items():
            print('\t%s: %s' % (k, v))

    #这里之前不占显存
    
    #pdb.set_trace()
    #加载模型大约7000MB的显存    
    if args.model == 'resnet18':
        assert args.dataset=='imagenet', 'resnet18 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=18, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == 'resnet34':
        assert args.dataset=='imagenet', 'resnet34 only supports imagenet dataset'
        net = ResNet_ImageNet(
            depth=34, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg)) 
    elif args.model == "resnet50":
        assert args.dataset == 'imagenet', 'resnet50 only supports imagenet dataset'
        #这里不知道什么操作会在0号卡上占3份100多M的显存
        net = ResNet_ImageNet(
            depth=50, num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "mobilenetv2":
        assert args.dataset == 'imagenet', 'mobilenetv2 only supports imagenet dataset'
        net = MobileNetV2(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg))
    elif args.model == "mobilenet":
        assert args.dataset == 'imagenet', 'mobilenet only supports imagenet dataset'
        net = MobileNet(
            num_classes=run_config.data_provider.n_classes, cfg=eval(args.cfg)) 
    else:
        print('model type not support')
    # print(net)
    # pdb.set_trace()
    # build run manager
    run_manager = RunManager(args.path, net, run_config) #这部分不会占显存

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    assert os.path.isfile(best_model_path), 'wrong path'
    if torch.cuda.is_available():
        # checkpoint = torch.load(best_model_path)
        checkpoint = torch.load(best_model_path, map_location='cpu') #如果用上面的写法，0号卡会多出3份checkpoint的显存，每份大约900M
    else:
        checkpoint = torch.load(best_model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    run_manager.net.load_state_dict(checkpoint)
    output_dict = {}   
    
    # feature extract
    data_loader = run_manager.run_config.test_loader
    data = next(iter(data_loader))
    data = data[0]
    n = data.size()[0]
    
    data = data.cuda()    
    
    with torch.no_grad():
        feature = net.feature_extract(data)   
    
    for i in range(len(feature)):
        feature[i] = feature[i].view(n, -1)
        feature[i] = feature[i].data.cpu().numpy()

    similar_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i]), cka.gram_linear(feature[j]))

    def sum_list(a, j):
        b = 0
        for i in range(len(a)):
            if i != j:
                b += a[i]
        return b

    important = []
    temp = []
    bops = []

    for i in range(len(feature)):
        temp.append( sum_list(similar_matrix[i], i) )

    b = sum_list(temp, -1) #list的总和
    temp = [x/b for x in temp] #除以总和

    for i in range(len(feature)):
        important.append( math.exp(-1* args.beta *temp[i] ) )
    # pdb.set_trace()

    length = len(net.cfg)
    init_quant_cfg = [32 for i in range(len(net.cfg))]
    bops_singlecfg, bops_doublecfg, bops_squarecfg = net.cfg2bops_perlayer(net.cfg, init_quant_cfg, length, args) #resnet中bops_squarecfg为0
    important = np.array(important)
    important = np.negative(important)#取相反数

    # Objective function
    def func(x, sign=1.0):
        """ Objective function """ 
        global important,length
        sum_fuc =[]
        for i in range(length):
            sum_fuc.append(x[i]*important[i]) #压缩率*重要性 不确定这里的压缩率要不要改成二次项
        return sum(sum_fuc)

    # Derivative function of objective function
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        global important
        diff = []
        for i in range(len(important)):
            diff.append(sign * (important[i]))
        return np.array(diff)

    # Constraint function
    def constrain_func(x):
        """ constrain function """
        global bops_singlecfg, bops_doublecfg, bops_squarecfg, length
        a = []
        for i in range(length):
            a.append(x[i] * bops_singlecfg[i])
            a.append(bops_squarecfg[i] * x[i] * x[i] * x[i])
        for i in range(1,length):
            for j in range(i):
                a.append(x[i] * x[i] * x[j] * bops_doublecfg[i][j]) #修改
        return np.array([args.target_bops - sum(a)]) #约束：target_bops-real_bops≥0

    def compute_cka_betw_model(init_model, model, data):
        data = data.cuda()
        model.cuda()
        init_model.cuda()
        n = data.size()[0]
        with torch.no_grad():
            init_features = init_model.feature_extract(data) #原网络features
            features = model.feature_extract(data)   #压缩后features
            for i in range(len(init_features)):
                init_features[i] = init_features[i].view(n, -1).data.cpu().numpy()   
                features[i] = features[i].view(n, -1).data.cpu().numpy() 
            sim_vec = np.zeros((len(features)))
            for i in range(len(init_features)):  
                sim_vec[i] = cka.cka(cka.gram_linear(init_features[i]), cka.gram_linear(features[i]))  
        return sim_vec  
    
    def compute_last_features_cka_betw_model(init_model, model, data):
        data = data.cuda()
        model.cuda()
        init_model.cuda()
        n = data.size()[0]
        with torch.no_grad():
            init_features = init_model.feature_extract(data) #原网络features
            features = model.feature_extract(data)   #压缩后features
            init_features = init_features[-1]
            features = features[-1]            
            init_features = init_features.view(n, -1).data.cpu().numpy()   
            features = features.view(n, -1).data.cpu().numpy()             
            sim_val = cka.cka(cka.gram_linear(init_features), cka.gram_linear(features))  
        return sim_val 

    bnds = []
    for i in range(length):
        bnds.append((0,1)) #len=37 [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1) ......]

    bnds = tuple(bnds) #len=37 ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1) ......)
    cons = ({'type': 'ineq',
             'fun': constrain_func})

    result = optimize.minimize(func,x0=[1 for i in range(length)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    # prun_cfg = np.around(np.array(net.cfg)*result.x)#每层压缩率*filter数=保留的filter数
    net.eval()
    cp_net = copy.deepcopy(net)
    cprate = result.x #np.array len = 37 for resnet50
    cprate = [x * x for x in cprate] #量化和剪枝的总压缩率
    bit_list = [4, 5, 6, 7, 8]

    prune_cfg = copy.deepcopy(net.cfg) #len = 37 for resnet50
    quant_cfg = [32 for i in range(len(net.cfg))]
    init_quant_cfg = copy.deepcopy(quant_cfg)
    cp_net.quantize_net(quant_cfg, args) #每层都量化为32bit   
    
    # pdb.set_trace()

    for i in range(length):  
        cp_net_list = []   
        cprate_cur_layer = cprate[i] #当前层的总cprate
        best_sim_val, best_index = 0, 0   
        
        for j, quantbit in enumerate(bit_list):                  
            quant_cprate = quantbit / 32  #当前层的量化压缩率
            # if quantbit != 8 and quant_cprate < cprate[i]: #如果量化bit比8低，且已经达到给定压缩率，跳过
            #     continue
            prune_cprate = cprate_cur_layer / (quant_cprate) #当前层的剪枝压缩率
            if prune_cprate > 1:
                prune_cprate = 1
                if args.local_rank == 0:
                    print('prune compression rate can not be larger than 1')            
            
            prune_cfg[i] = round(net.cfg[i] * prune_cprate)
            if prune_cfg[i] < net.cfg[i]/2:#增加
                prune_cfg[i] = net.cfg[i]//2
                if args.local_rank == 0:
                    print('remained channels can not be less than 1/2 of origin') 
                
            quant_cfg[i] = quantbit
            if args.model == 'resnet18':
                cp_net_list.append(ResNet_ImageNet(depth=18, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)) #生成小模型
            elif args.model == 'resnet34':
                cp_net_list.append(ResNet_ImageNet(depth=34, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)) #生成小模型
            elif args.model == 'resnet50':
                cp_net_list.append(ResNet_ImageNet(depth=50, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)) #生成小模型
            elif args.model == "mobilenet":
                cp_net_list.append(MobileNet(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg))
            elif args.model == "mobilenetv2":
                cp_net_list.append(MobileNetV2(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg))
            cp_net_list[j].init_model(run_config.model_init, run_config.init_div_groups)#对小模型进行初始化
            # cp_net_list[j].quantize_net(init_quant_cfg, args)             
            # get_unpruned_weights(cp_net_list[j], cp_net) #把模型权重根据L1范数大小迁移到小模型里
            get_unpruned_weights(cp_net_list[j], net)
            cp_net_list[j].quantize_net(quant_cfg, args)

            sim_vec = compute_cka_betw_model(net, cp_net_list[j], data)
            sim_val = np.sum(sim_vec)
            if sim_val > best_sim_val:
                best_index = j
                best_sim_val = sim_val
                best_prune_cfg = prune_cfg[i]
                best_quant_cfg = quant_cfg[i]

        cp_net = copy.deepcopy(cp_net_list[best_index])      
        prune_cfg[i] = best_prune_cfg
        quant_cfg[i] = best_quant_cfg #首尾层的不会计入
    if args.local_rank == 0:
        # pdb.set_trace()
        print('prune_cfg:',prune_cfg)
        print('quant_cfg:',quant_cfg)
    # print('剪枝后FLOPs', net.cfg2flops(prune_cfg))
    # print('原始FLOPs', net.cfg2flops(net.cfg))

    init_sum_bops = net.cfg2fpbops(net.cfg, length, args)
    if args.local_rank == 0:
        print('原始BOPs',init_sum_bops / 1e9)

    sum_bops = net.cfg2bops(prune_cfg, quant_cfg, length, args)    
    if args.local_rank == 0:
        print('压缩后BOPs',sum_bops / 1e9)
   
    while(sum_bops < args.target_bops):        
        best_sim_val, best_layer_index, flag = 0, 0, False  
        
        for i in range(length):
            if quant_cfg[i] < 8:
                new_quant_cfg = copy.deepcopy(quant_cfg)
                new_quant_cfg[i] += 1
                if args.model == "resnet18":
                    new_net = ResNet_ImageNet(depth=18, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet34":
                    new_net = ResNet_ImageNet(depth=34, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet50":
                    new_net = ResNet_ImageNet(depth=50, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenet":
                    new_net = MobileNet(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenetv2":
                    new_net = MobileNetV2(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                new_net.init_model(run_config.model_init, run_config.init_div_groups)#对小模型进行初始化
                get_unpruned_weights(new_net, net) #把模型权重根据L1范数大小迁移到小模型里
                new_net.quantize_net(new_quant_cfg, args)
                new_sim_vec = compute_cka_betw_model(net, new_net, data)
                new_quant_sim_val = np.sum(new_sim_vec)
                new_quant_bops = net.cfg2bops(prune_cfg, new_quant_cfg, length, args)
                #增加
                if new_quant_sim_val > best_sim_val:
                    flag = False
                    best_sim_val = new_quant_sim_val  
                    new_cfg = copy.deepcopy(new_quant_cfg)
                    sum_bops = new_quant_bops
        
        for i in range(length):#增加
            if prune_cfg[i] + round(net.cfg[i] * args.add_channel_rate) <= net.cfg[i]:
                new_prune_cfg = copy.deepcopy(prune_cfg)
                new_prune_cfg[i] += round(net.cfg[i] * args.add_channel_rate)
                if args.model == "resnet18":
                    new_net = ResNet_ImageNet(depth=18, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet34":
                    new_net = ResNet_ImageNet(depth=34, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet50":
                    new_net = ResNet_ImageNet(depth=50, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenet":
                    new_net = MobileNet(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenetv2":
                    new_net = MobileNetV2(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                new_net.init_model(run_config.model_init, run_config.init_div_groups)#对小模型进行初始化
                get_unpruned_weights(new_net, net) #把模型权重根据L1范数大小迁移到小模型里
                new_net.quantize_net(quant_cfg, args)
                new_sim_vec = compute_cka_betw_model(net, new_net, data)
                new_prune_sim_val = np.sum(new_sim_vec)
                new_prune_bops = net.cfg2bops(new_prune_cfg, quant_cfg, length, args)     
                #增加
                if new_prune_sim_val > best_sim_val:
                    flag = True
                    best_sim_val = new_prune_sim_val 
                    new_cfg = copy.deepcopy(new_prune_cfg)
                    sum_bops = new_prune_bops            

        # pdb.set_trace()
        if flag:#说明最终采用剪枝
            prune_cfg = copy.deepcopy(new_cfg)
            #quant_cfg不变
        else:#说明最终采用量化
            quant_cfg = copy.deepcopy(new_cfg)
            #prune_cfg不变
        if args.local_rank == 0:
            print('BOPs',sum_bops / 1e9)
            print('prune_cfg',prune_cfg)
            print('quant_cfg',quant_cfg)
    if args.local_rank == 0:
        print('last prune_cfg',prune_cfg)
        print('last quant_cfg',quant_cfg)
        print('last sum_bops',sum_bops/1e9)

    while(sum_bops > args.target_bops):        
        best_sim_val, best_layer_index, flag = 0, 0, False   
        for i in range(length):
            if quant_cfg[i] > 4:
                new_quant_cfg = copy.deepcopy(quant_cfg)
                new_quant_cfg[i] -= 1
                if args.model == "resnet18":
                    new_net = ResNet_ImageNet(depth=18, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet34":
                    new_net = ResNet_ImageNet(depth=34, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet50":
                    new_net = ResNet_ImageNet(depth=50, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenet":
                    new_net = MobileNet(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenetv2":
                    new_net = MobileNetV2(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                new_net.init_model(run_config.model_init, run_config.init_div_groups)#对小模型进行初始化
                get_unpruned_weights(new_net, net) #把模型权重根据L1范数大小迁移到小模型里
                new_net.quantize_net(new_quant_cfg, args)
                new_sim_vec = compute_cka_betw_model(net, new_net, data)
                new_quant_sim_val = np.sum(new_sim_vec)
                new_quant_bops = net.cfg2bops(prune_cfg, new_quant_cfg, length, args)

                if new_quant_sim_val > best_sim_val:
                    flag = False
                    best_sim_val = new_quant_sim_val  
                    new_cfg = copy.deepcopy(new_quant_cfg)
                    sum_bops = new_quant_bops

        for i in range(length):
            if prune_cfg[i] - round(net.cfg[i] * args.add_channel_rate) >= net.cfg[i]//2:
                new_prune_cfg = copy.deepcopy(prune_cfg)
                new_prune_cfg[i] -= round(net.cfg[i] * args.add_channel_rate)
                if args.model == "resnet18":
                    new_net = ResNet_ImageNet(depth=18, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet34":
                    new_net = ResNet_ImageNet(depth=34, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "resnet50":
                    new_net = ResNet_ImageNet(depth=50, num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenet":
                    new_net = MobileNet(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                elif args.model == "mobilenetv2":
                    new_net = MobileNetV2(num_classes=run_config.data_provider.n_classes, cfg=prune_cfg)
                new_net.init_model(run_config.model_init, run_config.init_div_groups)#对小模型进行初始化
                get_unpruned_weights(new_net, net) #把模型权重根据L1范数大小迁移到小模型里
                new_net.quantize_net(quant_cfg, args)
                new_sim_vec = compute_cka_betw_model(net, new_net, data)
                new_prune_sim_val = np.sum(new_sim_vec)
                new_prune_bops = net.cfg2bops(new_prune_cfg, quant_cfg, length, args)  

                if new_prune_sim_val > best_sim_val:
                    flag = True
                    best_sim_val = new_prune_sim_val 
                    new_cfg = copy.deepcopy(new_prune_cfg)
                    sum_bops = new_prune_bops                    

        if flag:#说明最终采用剪枝
            prune_cfg = copy.deepcopy(new_cfg)
            #quant_cfg不变
        else:#说明最终采用量化
            quant_cfg = copy.deepcopy(new_cfg)
            #prune_cfg不变
        if args.local_rank == 0:
            print('BOPs',sum_bops / 1e9)
            print('prune_cfg',prune_cfg)
            print('quant_cfg',quant_cfg)        
    if args.local_rank == 0:
        print('last prune_cfg',prune_cfg)
        print('last quant_cfg',quant_cfg)
        print('last sum_bops',sum_bops/1e9)
    
