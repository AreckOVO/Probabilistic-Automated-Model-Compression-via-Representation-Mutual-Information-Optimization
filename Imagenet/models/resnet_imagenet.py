import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.base_models import *
from collections import OrderedDict
import sys
import numpy as np
from utils import quantize
from utils.quantize import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2=0, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv3x3(inplanes, planes_1, stride)
        bn1 = norm_layer(planes_1)
        relu = nn.ReLU(inplace=True)
        if planes_2 == 0: #每个stage非第一个block的第二个conv不用手动指定输出通道，只要使用上一个block最后一个conv的输出通道
            conv2 = conv3x3(planes_1, inplanes)
            bn2 = norm_layer(inplanes)
        else: #每个stage第一个block的第二个conv要手动指定输出通道
            conv2 = conv3x3(planes_1, planes_2)
            bn2 = norm_layer(planes_2)
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2, planes_3=0, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv1x1(inplanes, planes_1)
        bn1 = norm_layer(planes_1)
        conv2 = conv3x3(planes_1, planes_2, stride)
        bn2 = norm_layer(planes_2)
        if planes_3 == 0:
            conv3 = conv1x1(planes_2, inplanes)
            bn3 = norm_layer(inplanes)
        else:
            conv3 = conv1x1(planes_2, planes_3)
            bn3 = norm_layer(planes_3)
        relu = nn.ReLU(inplace=True)
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2), ('relu', relu)]))
        self.conv3 = nn.Sequential(OrderedDict([('conv', conv3), ('bn', bn3)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_ImageNet(MyNetwork):
    def __init__(self, cfg=None, depth=18, block=BasicBlock, num_classes=1000):
        super(ResNet_ImageNet, self).__init__()
        #对于resnet18和34，传入初始block的out_channel，在每个stage内，对于第一个block，需要手动传入2个cfg，分别作为第一个和第二个conv的输出通道，对于其他block只要传入一个cfg作为第一个conv的输出通道，
        # 由于block的shortcut element-wise add，导致这些block内第二个conv的输出通道是固定的，因此不用手动指定，除此之外，上一个conv的输出通道和下一个conv的输入通道一一对应
        #所以对于对于resnet18和34，每个stage内第一个block要传入2个cfg，其他都只要传入1个cfg
        self.cfgs_base = {18: [64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
                          34: [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512],
                          50: [64, 64, 64, 256, 64, 64, 64, 64, 128, 128, 512, 128, 128, 128, 128, 128, 128, 256, 256, 1024, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 2048, 512, 512, 512, 512]}
        #对于resnet50，传入初始block的out_channel，在每个stage内，对于第一个block，需要手动传入3个cfg，分别作为第1、2、3个conv的输出通道，对于其他block只要传入一个cfg作为第一个和第二个conv的输出通道，
        # 由于block的shortcut element-wise add，导致这些block内第3个conv的输出通道是固定的，因此不用手动指定，除此之外，上一个conv的输出通道和下一个conv的输入通道一一对应        
        #所以对于对于resnet50，每个stage内第一个block要传入3个cfg，其他都只要传入2个cfg
        if depth==18:
            block = BasicBlock
            blocks = [2, 2, 2, 2]
            _cfg = self.cfgs_base[18] #基础的resnet18每层的out_channel
        elif depth==34:
            block = BasicBlock
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[34] #基础的resnet34每层的out_channel
        elif depth==50:
            block = Bottleneck
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[50] #基础的resnet50每层的out_channel
        if cfg == None:
            cfg = _cfg #如果创建网络的时候，没有传入额外的cfg，那么就按照cfgs_base里的out_channel来生成（不剪枝）
        norm_layer = nn.BatchNorm2d
        self.num_classes = num_classes
        self._norm_layer = norm_layer
        self.depth = depth
        self.cfg = cfg
        self.inplanes = cfg[0] #第一个block的conv的输出通道
        self.blocks = blocks
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),#第一个block的conv的输出通道
                                                ('bn', norm_layer(self.inplanes)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if depth!=50: #resnet18、34用的是basicblock
            self.layer1 = self._make_layer(block, cfg[1 : blocks[0]+2], blocks[0]) #1 : blocks[0]+2表示只取到blocks[0]+1 对于resnet18， 1~3
            self.layer2 = self._make_layer(block, cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], blocks[1], stride=2,) #4~6
            self.layer3 = self._make_layer(block, cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], blocks[2], stride=2,) #7~9
            self.layer4 = self._make_layer(block, cfg[blocks[0]+blocks[1]+blocks[2]+4: ], blocks[3], stride=2,) #10~12
            self.fc = nn.Linear(cfg[blocks[0]+blocks[1]+blocks[2]+5], num_classes) #11
        else: #resnet50用的是bottleneck
            self.layer1 = self._make_layer(block, cfg[1 : 2*blocks[0]+2], blocks[0])
            self.layer2 = self._make_layer(block, cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1], blocks[1], stride=2,)
            self.layer3 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4], blocks[2], stride=2,)
            self.layer4 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ], blocks[3], stride=2,)
            self.fc = nn.Linear(cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+6], num_classes)            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        #block: block的类型，包括BasicBlock和Bottleneck
        #planes: channel
        #blocks: 该stage有几个block
        norm_layer = self._norm_layer
        downsample = None
        if self.depth == 50:
            first_planes = planes[0:3] #0:3表示只取0,1,2
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)),
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], first_planes[2], stride, downsample, norm_layer)) #resnet50用bottleneck有3个conv
            self.inplanes = first_planes[-1]
            later_planes = planes[3:3+2*(blocks-1)]
            for i in range(1, blocks):
                layers.append(block(self.inplanes, later_planes[2*(i-1)], later_planes[2*(i-1)+1], norm_layer=norm_layer))
            return nn.Sequential(*layers)
        else:
            first_planes = planes[0:2] #0:2表示只取0和1
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)), #第一个stage时，self.inplanes=cfg[0]
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], stride, downsample, norm_layer)) 
            #参数分别为conv1的in_channel，conv1的out_channel==conv2的in_channel,conv2的out_channel
            self.inplanes = first_planes[-1] #conv2的out_channel作为下一个block的in_channel
            later_planes = planes[2:2+blocks-1] #除了stage内第一个block，后面每个block都只需要一个cfg
            for i in range(1, blocks):#遍历stage内第一个后的所有block
                layers.append(block(self.inplanes, later_planes[i-1], norm_layer=norm_layer)) #除了stage内第一个block，后面每个block都只传入一个cfg
                #conv1的in_channel,conv1的out_channel==conv2的in_channel，没有第三个参数的话表示conv2的out_channel等于self.inplanes，意思是说stage内的后几个block的in_channel和out_channel一样
            return nn.Sequential(*layers)
    
    def quantize_net(self, quant_cfg, args):
        for i in range(len(quant_cfg)):
            self.quantize_layer(quant_cfg[i], i, args)
        self.fc = quantize_fc(self.fc, 32, args)
        self.fc.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
        self.fc.activation_quantizer.bitwidth_refactor(refactored_bit = 32)
        
    def quantize_layer(self, w_bits, idx, args):
        blocks = self.blocks
        if idx == 0:           
            self.conv1.conv = quantize_conv(self.conv1.conv, 32, args)#第一层量化为32bit
            self.conv1.conv.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
            self.conv1.conv.activation_quantizer.bitwidth_refactor(refactored_bit = 32)            
            return
            
        if self.depth!=50:  #resnet18 34
            if idx in range(1, blocks[0]+2):  #stage1
                if (idx - 1) < 2:#量化第一个block的conv 1 2
                    # setattr(self.layer1[0],'conv{}'.format(idx - 1 + 1), quantize_conv(getattr(self.layer1[0],'conv{}'.format(idx - 1 + 1)), w_bits, args))
                    tmpconv = getattr(self.layer1[0],'conv{}'.format(idx - 1 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - 1) == 1:#如果量化的是block1的conv2，同时量化block1的downsample
                        self.layer1[0].downsample.conv = quantize_conv(self.layer1[0].downsample.conv, w_bits, args)
                else: #量化其他block的conv1
                    self.layer1[(idx - 1 - 2) + 1].conv1.conv = quantize_conv(self.layer1[(idx - 1 - 2) + 1].conv1.conv, w_bits, args)
                    tmp_w_bits = self.layer1[0].conv2.conv.weight_quantizer.bits
                    self.layer1[(idx - 1 - 2) + 1].conv2.conv = quantize_conv(self.layer1[(idx - 1 - 2) + 1].conv2.conv, tmp_w_bits, args) #量化conv1的同时也量化conv2

            elif idx in range(blocks[0]+2, blocks[0]+2+blocks[1]+1):   #stage2
                if (idx - (blocks[0]+2)) < 2:
                    # setattr(self.layer2[0],'conv{}'.format(idx - (blocks[0]+2) + 1), quantize_conv(getattr(self.layer2[0],'conv{}'.format(idx - (blocks[0]+2) + 1)), w_bits, args))
                    tmpconv = getattr(self.layer2[0],'conv{}'.format(idx - (blocks[0]+2) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - (blocks[0]+2)) == 1:
                        self.layer2[0].downsample.conv = quantize_conv(self.layer2[0].downsample.conv, w_bits, args)
                else:
                    self.layer2[(idx - (blocks[0]+2) - 2) + 1].conv1.conv = quantize_conv(self.layer2[(idx - (blocks[0]+2) - 2) + 1].conv1.conv, w_bits, args)
                    tmp_w_bits = self.layer2[0].conv2.conv.weight_quantizer.bits
                    self.layer2[(idx - (blocks[0]+2) - 2) + 1].conv2.conv = quantize_conv(self.layer2[(idx - (blocks[0]+2) - 2) + 1].conv2.conv, tmp_w_bits, args) 
                    
            elif idx in range(blocks[0]+blocks[1]+3, blocks[0]+blocks[1]+blocks[2]+4):   #stage3
                if (idx - (blocks[0]+blocks[1]+3)) < 2:
                    # setattr(self.layer3[0],'conv{}'.format(idx - (blocks[0]+blocks[1]+3) + 1), quantize_conv(getattr(self.layer3[0],'conv{}'.format(idx - (blocks[0]+blocks[1]+3) + 1)), w_bits, args))
                    tmpconv = getattr(self.layer3[0],'conv{}'.format(idx - (blocks[0]+blocks[1]+3) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - (blocks[0]+blocks[1]+3)) == 1:
                        self.layer3[0].downsample.conv = quantize_conv(self.layer3[0].downsample.conv, w_bits, args)
                else:
                    self.layer3[(idx - (blocks[0]+blocks[1]+3) - 2) + 1].conv1.conv = quantize_conv(self.layer3[(idx - (blocks[0]+blocks[1]+3) - 2) + 1].conv1.conv, w_bits, args)
                    tmp_w_bits = self.layer3[0].conv2.conv.weight_quantizer.bits
                    self.layer3[(idx - (blocks[0]+blocks[1]+3) - 2) + 1].conv2.conv = quantize_conv(self.layer3[(idx - (blocks[0]+blocks[1]+3) - 2) + 1].conv2.conv, tmp_w_bits, args) 

            elif idx in range(blocks[0]+blocks[1]+blocks[2]+4, blocks[0]+blocks[1]+blocks[2]+blocks[3]+5):   #stage4
                if (idx - (blocks[0]+blocks[1]+blocks[2]+4)) < 2:
                    # setattr(self.layer4[0], 'conv{}'.format(idx - (blocks[0]+blocks[1]+blocks[2]+4) + 1), quantize_conv(getattr(self.layer4[0],'conv{}'.format(idx - (blocks[0]+blocks[1]+blocks[2]+4) + 1)), w_bits, args))
                    tmpconv = getattr(self.layer4[0],'conv{}'.format(idx - (blocks[0]+blocks[1]+blocks[2]+4) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - (blocks[0]+blocks[1]+blocks[2]+4)) == 1:
                        self.layer4[0].downsample.conv = quantize_conv(self.layer4[0].downsample.conv, w_bits, args)
                else:
                    self.layer4[(idx - (blocks[0]+blocks[1]+blocks[2]+4) - 2) + 1].conv1.conv = quantize_conv(self.layer4[(idx - (blocks[0]+blocks[1]+blocks[2]+4) - 2) + 1].conv1.conv, w_bits, args)
                    tmp_w_bits = self.layer4[0].conv2.conv.weight_quantizer.bits
                    self.layer4[(idx - (blocks[0]+blocks[1]+blocks[2]+4) - 2) + 1].conv2.conv = quantize_conv(self.layer4[(idx - (blocks[0]+blocks[1]+blocks[2]+4) - 2) + 1].conv2.conv, tmp_w_bits, args)
            else:
                print('quantize idx error!')
        else:  #resnet50
            if idx in range(1, 2*blocks[0]+2):  #stage1
                if (idx - 1) < 3:  
                    tmpconv = getattr(self.layer1[0],'conv{}'.format(idx - 1 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - 1) == 2:#如果量化的是conv3，同时量化block1的downsample
                        self.layer1[0].downsample.conv = quantize_conv(self.layer1[0].downsample.conv, w_bits, args)
                        # for i in range(1, blocks[0]):
                        #     self.layer1[i].conv3.conv = quantize_conv(self.layer1[i].conv3.conv, w_bits, args)
                else: #量化其他block的conv 1 2                    
                    tmpconv = getattr(self.layer1[(idx - 1 - 3) // 2 + 1],'conv{}'.format((idx - 1 - 3) % 2 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if ((idx - 1 - 3) % 2 + 1) == 2: #量化conv2的同时也量化conv3
                        tmp_w_bits = self.layer1[0].conv3.conv.weight_quantizer.bits
                        self.layer1[(idx - 1 - 3) // 2 + 1].conv3.conv = quantize_conv(self.layer1[(idx - 1 - 3) // 2 + 1].conv3.conv, tmp_w_bits, args) 

            elif idx in range(2*blocks[0]+2, 2*blocks[0]+2+2*blocks[1]+1):   #stage2
                if (idx - (2*blocks[0]+2)) < 3:                    
                    tmpconv = getattr(self.layer2[0],'conv{}'.format(idx - (2*blocks[0]+2) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - (2*blocks[0]+2)) == 2: 
                        self.layer2[0].downsample.conv = quantize_conv(self.layer2[0].downsample.conv, w_bits, args)
                        # for i in range(1, blocks[1]):
                        #     self.layer2[i].conv3.conv = quantize_conv(self.layer2[i].conv3.conv, w_bits, args)
                else:                    
                    tmpconv = getattr(self.layer2[(idx - (2*blocks[0]+2) - 3) // 2 + 1],'conv{}'.format((idx - (2*blocks[0]+2) - 3) % 2 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if ((idx - (2*blocks[0]+2) - 3) % 2 + 1) == 2:
                        tmp_w_bits = self.layer2[0].conv3.conv.weight_quantizer.bits
                        self.layer2[(idx - (2*blocks[0]+2) - 3) // 2 + 1].conv3.conv = quantize_conv(self.layer2[(idx - (2*blocks[0]+2) - 3) // 2 + 1].conv3.conv, tmp_w_bits, args) 
                        
            elif idx in range(2*blocks[0]+2*blocks[1]+3, 2*blocks[0]+2*blocks[1]+2*blocks[2]+4):   #stage3
                if (idx - (2*blocks[0]+2*blocks[1]+3)) < 3:                    
                    tmpconv = getattr(self.layer3[0],'conv{}'.format(idx - (2*blocks[0]+2*blocks[1]+3) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if (idx - (2*blocks[0]+2*blocks[1]+3)) == 2:
                        self.layer3[0].downsample.conv = quantize_conv(self.layer3[0].downsample.conv, w_bits, args)
                        # for i in range(1, blocks[2]):
                        #     self.layer3[i].conv3.conv = quantize_conv(self.layer3[i].conv3.conv, w_bits, args)
                else:                    
                    tmpconv = getattr(self.layer3[(idx - (2*blocks[0]+2*blocks[1]+3) - 3) // 2 + 1],'conv{}'.format((idx - (2*blocks[0]+2*blocks[1]+3) - 3) % 2 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if ((idx - (2*blocks[0]+2*blocks[1]+3) - 3) % 2 + 1) == 2:
                        tmp_w_bits = self.layer3[0].conv3.conv.weight_quantizer.bits
                        self.layer3[(idx - (2*blocks[0]+2*blocks[1]+3) - 3) // 2 + 1].conv3.conv = quantize_conv(self.layer3[(idx - (2*blocks[0]+2*blocks[1]+3) - 3) // 2 + 1].conv3.conv, tmp_w_bits, args) 

            elif idx in range(2*blocks[0]+2*blocks[1]+2*blocks[2]+4, 2*blocks[0]+2*blocks[1]+2*blocks[2]+2*blocks[3]+5):   #stage4
                if (idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4)) < 3:
                    tmpconv = getattr(self.layer4[0],'conv{}'.format(idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)                    
                    if (idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4)) == 2:
                        self.layer4[0].downsample.conv = quantize_conv(self.layer4[0].downsample.conv, w_bits, args)
                        # for i in range(1, blocks[3]):
                        #     self.layer4[i].conv3.conv = quantize_conv(self.layer4[i].conv3.conv, w_bits, args)
                else:                   
                    tmpconv = getattr(self.layer4[(idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) - 3) // 2 + 1],'conv{}'.format((idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) - 3) % 2 + 1))
                    tmpconv.conv = quantize_conv(tmpconv.conv, w_bits, args)
                    if ((idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) - 3) % 2 + 1) == 2:
                        tmp_w_bits = self.layer4[0].conv3.conv.weight_quantizer.bits
                        self.layer4[(idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) - 3) // 2 + 1].conv3.conv = quantize_conv(self.layer4[(idx - (2*blocks[0]+2*blocks[1]+2*blocks[2]+4) - 3) // 2 + 1].conv3.conv, tmp_w_bits, args) 
            else:
                print('quantize idx error!')

    def cfg2params(self, cfg):
        #通过cfg计算整个网络的参数量
        blocks = self.blocks
        params = 0.
        params += (3 * 7 * 7 * cfg[0] + 2 * cfg[0]) # first layer，2是bn参数
        inplanes = cfg[0]
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2],
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], 
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4],
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]]
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4):
            planes = sub_cfgs[i] #planes代表每个stage的out_channel，它本身就是个list，存放着stage中每个block的channel
            if self.depth != 50:
                first_planes = planes[0:2] #stage内第一个block的out_channel list
                later_planes = planes[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3] #stage内第一个block的out_channel list
                later_planes = planes[3:3+2*(blocks[i]-1)]
            params += (inplanes * 1 * 1 * first_planes[-1] + 2 * first_planes[-1]) # downsample layer
            if self.depth != 50:
                params += (inplanes * 3 * 3 * first_planes[0] + 2 * first_planes[0])
                params += (first_planes[0] * 3 * 3 * first_planes[1] + 2 * first_planes[1])
            else:
                params += (inplanes * 1 * 1 * first_planes[0] + 2 * first_planes[0])
                params += (first_planes[0] * 3 * 3 * first_planes[1] + 2 * first_planes[1])
                params += (first_planes[1] * 1 * 1 * first_planes[2] + 2 * first_planes[2])
            for j in range(1, self.blocks[i]):
                inplanes = first_planes[-1]
                if self.depth != 50:
                    params += (inplanes * 3 * 3 * later_planes[j-1] + 2 * later_planes[j-1])
                    params += (later_planes[j-1] * 3 * 3 * inplanes + 2 * inplanes)
                else:
                    params += (inplanes * 1 * 1 * later_planes[2*(j-1)] + 2 * later_planes[2*(j-1)])
                    params += (later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] + 2 * later_planes[2*(j-1)+1])
                    params += (later_planes[2*(j-1)+1] * 1 * 1 * inplanes + 2 * inplanes)
        if self.depth==50:
            params += (cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+6] + 1) * self.num_classes
        else:
            params += (cfg[blocks[0]+blocks[1]+blocks[2]+5] + 1) * self.num_classes
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        #通过cfg计算整个网络的flops，只要修改网络的self.cfg = cfg就行，不用修改其他参数
        blocks = self.blocks
        flops = 0.
        size = 224
        size /= 2 # first conv layer s=2
        flops += (3 * 7 * 7 * cfg[0] * size * size + 5 * cfg[0] * size * size) # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2 # pooling s=2
        flops += (3 * 3 * cfg[0] * size * size) # maxpooling
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2],
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], 
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4],
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]]
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2] #stage内第一个block的比特数，需要2个
                later_planes = planes[2:2+blocks[i]-1] #stage内后面所有的block
            else:
                first_planes = planes[0:3] #stage内第一个block的比特数，需要3个
                later_planes = planes[3:3+2*(blocks[i]-1)] #stage内后面所有的block
            if i in [1, 2, 3]:
                size /= 2
            flops += (inplanes * 1 * 1 * first_planes[-1] + 5 * first_planes[-1]) * size * size # downsample layer
            
            if self.depth != 50: #stage内第一个block
                flops += (inplanes * 3 * 3 * first_planes[0] + 5 * first_planes[0]) * size * size #conv1
                flops += (first_planes[0] * 3 * 3 * first_planes[1] + 5 * first_planes[1]) * size * size #conv2
            else:
                size *= 2
                flops += (inplanes * 1 * 1 * first_planes[0] + 5 * first_planes[0]) * size * size #con1
                size /= 2
                flops += (first_planes[0] * 3 * 3 * first_planes[1] + 5 * first_planes[1]) * size * size #conv2
                flops += (first_planes[1] * 1 * 1 * first_planes[2] + 5 * first_planes[2]) * size * size #conv3
            for j in range(1, self.blocks[i]): #后几个block
                inplanes = first_planes[-1]
                if self.depth != 50:
                    flops += (inplanes * 3 * 3 * later_planes[j-1] + 5 * later_planes[j-1]) * size * size #conv1
                    flops += (later_planes[j-1] * 3 * 3 * inplanes + 5 * inplanes) * size * size #conv2
                else:
                    flops += (inplanes * 1 * 1 * later_planes[2*(j-1)] + 5 * later_planes[2*(j-1)]) * size * size #conv1
                    flops += (later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] + 5 * later_planes[2*(j-1)+1]) * size * size #conv2
                    flops += (later_planes[2*(j-1)+1] * 1 * 1 * inplanes + 5 * inplanes) * size * size #conv3
        flops += (2 * cfg[-1] + 1) * self.num_classes
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        #通过cfg，计算每层的flops，length是cfg的长度，只要修改网络的self.cfg就行，不用修改其他参数
        blocks = self.blocks
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]
        size = 224
        size /= 2 # first conv layer s=2
        flops_singlecfg[0] += (3 * 7 * 7 * cfg[0] * size * size + 5 * cfg[0] * size * size) # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2 # pooling s=2
        flops_singlecfg[0] += (3 * 3 * cfg[0] * size * size) # maxpooling
        if self.depth != 50:
            count = 2
        else:
            count = 3
        count1 = 0
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2], #stage1
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2] #第一个block
                later_planes = planes[2:2+blocks[i]-1] #stage内后面所有的block
            else:
                first_planes = planes[0:3] #第一个block
                later_planes = planes[3:3+2*(blocks[i]-1)] #stage内后面所有的block
            if i in [1, 2, 3]: #第一个stage不做downsample
                size /= 2
                if self.depth != 50:
                    count += blocks[i-1]+1 #这里count的位置在下一个stage第一个block的最后一个conv
                else:
                    count += 2 * blocks[i-1] + 1
            flops_doublecfg[count][count1] += inplanes * 1 * 1 * first_planes[-1] * size * size # downsample layer
            flops_doublecfg[count1][count] += inplanes * 1 * 1 * first_planes[-1] * size * size
            flops_singlecfg[count] += 5 * first_planes[-1] * size * size
            
            if self.depth != 50:
                flops_doublecfg[count-1][count1] += inplanes * 3 * 3 * first_planes[0] * size * size #conv1
                flops_doublecfg[count1][count-1] += inplanes * 3 * 3 * first_planes[0] * size * size
                flops_singlecfg[count-1] += 5 * first_planes[0] * size * size

                flops_doublecfg[count-1][count] += first_planes[0] * 3 * 3 * first_planes[1] * size * size #conv2
                flops_doublecfg[count][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_singlecfg[count] += 5 * first_planes[1] * size * size
            else:
                size *= 2
                flops_doublecfg[count-2][count1] += inplanes * 1 * 1 * first_planes[0] * size * size #conv1
                flops_doublecfg[count1][count-2] += inplanes * 1 * 1 * first_planes[0] * size * size
                flops_singlecfg[count-2] += 5 * first_planes[0] * size * size

                size /= 2
                flops_doublecfg[count-2][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size #conv2
                flops_doublecfg[count-1][count-2] += first_planes[0] * 3 * 3 * first_planes[1] * size * size
                flops_singlecfg[count-1] += 5 * first_planes[1] * size * size

                flops_doublecfg[count][count-1] += first_planes[1] * 1 * 1 * first_planes[2] * size * size #conv3
                flops_doublecfg[count-1][count] += first_planes[1] * 1 * 1 * first_planes[2] * size * size
                flops_singlecfg[count] += 5 * first_planes[2] * size * size

            for j in range(1, self.blocks[i]): #stage内后几个block
                inplanes = first_planes[-1]
                count1 = count
                if self.depth != 50:
                    flops_doublecfg[count1][count+j] += inplanes * 3 * 3 * later_planes[j-1] * size * size
                    flops_doublecfg[count+j][count1] += inplanes * 3 * 3 * later_planes[j-1] * size * size
                    flops_singlecfg[count+j] += 5 * later_planes[j-1] * size * size

                    flops_doublecfg[count1][count+j] += later_planes[j-1] * 3 * 3 * inplanes * size * size
                    flops_doublecfg[count+j][count1] += later_planes[j-1] * 3 * 3 * inplanes * size * size
                    flops_singlecfg[count1] += 5 * inplanes * size * size
                else:
                    flops_doublecfg[count1][count+1+2*(j-1)] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size
                    flops_doublecfg[count+1+2*(j-1)][count1] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size
                    flops_singlecfg[count+1+2*(j-1)] += 5 * later_planes[2*(j-1)] * size * size

                    flops_doublecfg[count+2+2*(j-1)][count+1+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size
                    flops_doublecfg[count+1+2*(j-1)][count+2+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size
                    flops_singlecfg[count+2+2*(j-1)] += 5 * later_planes[2*(j-1)+1] * size * size

                    flops_doublecfg[count+2+2*(j-1)][count1] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size
                    flops_doublecfg[count1][count+2+2*(j-1)] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size
                    flops_singlecfg[count1] += 5 * inplanes * size * size
        flops_singlecfg[-1] += (2 * cfg[-1] + 1) * self.num_classes
        return flops_singlecfg, flops_doublecfg, flops_squarecfg

    """"CAL BITOPS"""
    """"CAL BITOPS"""
    def cfg2bops(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution bops
        #通过cfg和quant_cfg，计算总bops，length是cfg的长度
        blocks = self.blocks
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = 224
        size /= 2 # first conv layer s=2
        bops_singlecfg[0] += (3 * 7 * 7 * cfg[0] * size * size * (quant_cfg[0] * args.a_bits) + 5 * cfg[0] * size * size * args.a_bits) # first layer, conv+bn+relu
        inplanes = cfg[0]
        inbits = quant_cfg[0]
        size /= 2 # pooling s=2
        bops_singlecfg[0] += (3 * 3 * cfg[0] * size * size * args.a_bits) # maxpooling
        if self.depth != 50:
            count = 2
        else:
            count = 3
        count1 = 0
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2], #stage1
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
            sub_quant_cfg = [quant_cfg[1 : blocks[0]+2], #stage1
                             quant_cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                             quant_cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                             quant_cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
            sub_quant_cfg = [quant_cfg[1 : 2*blocks[0]+2],
                             quant_cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                             quant_cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                             quant_cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            bits = sub_quant_cfg[i]
            if self.depth != 50:
                first_planes = planes[0:2] #第一个block
                later_planes = planes[2:2+blocks[i]-1] #stage内后面所有的block
                first_bits = bits[0:2]
                later_bits = bits[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3] #第一个block
                later_planes = planes[3:3+2*(blocks[i]-1)] #stage内后面所有的block
                first_bits = bits[0:3] 
                later_bits = bits[3:3+2*(blocks[i]-1)]
            if i in [1, 2, 3]: #第一个stage不做downsample
                size /= 2
                if self.depth != 50:
                    count += blocks[i-1]+1 #这里count的位置在下一个stage第一个block的最后一个conv
                else:
                    count += 2 * blocks[i-1] + 1
            bops_doublecfg[count][count1] += inplanes * 1 * 1 * first_planes[-1] * size * size * (first_bits[-1] * args.quant_a_bits) # downsample layer
            bops_doublecfg[count1][count] += inplanes * 1 * 1 * first_planes[-1] * size * size * (first_bits[-1] * args.quant_a_bits) 
            bops_singlecfg[count] += 5 * first_planes[-1] * size * size * (args.quant_a_bits) 
            
            if self.depth != 50:
                bops_doublecfg[count-1][count1] += inplanes * 3 * 3 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits) #conv1
                bops_doublecfg[count1][count-1] += inplanes * 3 * 3 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)
                bops_singlecfg[count-1] += 5 * first_planes[0] * size * size * (args.quant_a_bits) 

                bops_doublecfg[count-1][count] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)  #conv2
                bops_doublecfg[count][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)  
                bops_singlecfg[count] += 5 * first_planes[1] * size * size * (args.quant_a_bits) 
            else:
                size *= 2
                bops_doublecfg[count-2][count1] += inplanes * 1 * 1 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)   #conv1
                bops_doublecfg[count1][count-2] += inplanes * 1 * 1 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)   
                bops_singlecfg[count-2] += 5 * first_planes[0] * size * size * (args.quant_a_bits) 

                size /= 2
                bops_doublecfg[count-2][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)   #conv2
                bops_doublecfg[count-1][count-2] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits) 
                bops_singlecfg[count-1] += 5 * first_planes[1] * size * size * (args.quant_a_bits) 

                bops_doublecfg[count][count-1] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (first_bits[2] * args.quant_a_bits)    #conv3
                bops_doublecfg[count-1][count] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (first_bits[2] * args.quant_a_bits) 
                bops_singlecfg[count] += 5 * first_planes[2] * size * size * (args.quant_a_bits) 

            for j in range(1, self.blocks[i]): #stage内后几个block
                inplanes = first_planes[-1]
                inbits = first_bits[-1]
                count1 = count
                if self.depth != 50:
                    bops_doublecfg[count1][count+j] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (later_bits[j-1] * args.quant_a_bits)
                    bops_doublecfg[count+j][count1] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (later_bits[j-1] * args.quant_a_bits)
                    bops_singlecfg[count+j] += 5 * later_planes[j-1] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count1][count+j] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_doublecfg[count+j][count1] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (args.quant_a_bits) 
                else:
                    bops_doublecfg[count1][count+1+2*(j-1)] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (later_bits[2*(j-1)] * args.quant_a_bits)
                    bops_doublecfg[count+1+2*(j-1)][count1] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (later_bits[2*(j-1)] * args.quant_a_bits)
                    bops_singlecfg[count+1+2*(j-1)] += 5 * later_planes[2*(j-1)] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count+2+2*(j-1)][count+1+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (later_bits[2*(j-1)+1] * args.quant_a_bits)
                    bops_doublecfg[count+1+2*(j-1)][count+2+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (later_bits[2*(j-1)+1] * args.quant_a_bits)
                    bops_singlecfg[count+2+2*(j-1)] += 5 * later_planes[2*(j-1)+1] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count+2+2*(j-1)][count1] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_doublecfg[count1][count+2+2*(j-1)] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (args.quant_a_bits) 
        bops_singlecfg[-1] += (2 * cfg[-1] + 1) * self.num_classes * args.a_bits * inbits
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops

    def cfg2bops_perlayer(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution bops
        #通过cfg和quant_cfg，计算每层的bops，length是cfg的长度
        blocks = self.blocks
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = 224
        size /= 2 # first conv layer s=2
        bops_singlecfg[0] += (3 * 7 * 7 * cfg[0] * size * size * (quant_cfg[0] * args.a_bits) + 5 * cfg[0] * size * size * args.a_bits) # first layer, conv+bn+relu
        inplanes = cfg[0]
        inbits = quant_cfg[0]
        size /= 2 # pooling s=2
        bops_singlecfg[0] += (3 * 3 * cfg[0] * size * size * args.a_bits) # maxpooling
        if self.depth != 50:
            count = 2
        else:
            count = 3
        count1 = 0
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2], #stage1
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
            sub_quant_cfg = [quant_cfg[1 : blocks[0]+2], #stage1
                             quant_cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                             quant_cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                             quant_cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
            sub_quant_cfg = [quant_cfg[1 : 2*blocks[0]+2],
                             quant_cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                             quant_cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                             quant_cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            bits = sub_quant_cfg[i]
            if self.depth != 50:
                first_planes = planes[0:2] #第一个block
                later_planes = planes[2:2+blocks[i]-1] #stage内后面所有的block
                first_bits = bits[0:2]
                later_bits = bits[2:2+blocks[i]-1]
            else:
                first_planes = planes[0:3] #第一个block
                later_planes = planes[3:3+2*(blocks[i]-1)] #stage内后面所有的block
                first_bits = bits[0:3] 
                later_bits = bits[3:3+2*(blocks[i]-1)]
            if i in [1, 2, 3]: #第一个stage不做downsample
                size /= 2
                if self.depth != 50:
                    count += blocks[i-1]+1 #这里count的位置在下一个stage第一个block的最后一个conv
                else:
                    count += 2 * blocks[i-1] + 1
            bops_doublecfg[count][count1] += inplanes * 1 * 1 * first_planes[-1] * size * size * (first_bits[-1] * args.quant_a_bits) # downsample layer
            bops_doublecfg[count1][count] += inplanes * 1 * 1 * first_planes[-1] * size * size * (first_bits[-1] * args.quant_a_bits) 
            bops_singlecfg[count] += 5 * first_planes[-1] * size * size * (args.quant_a_bits) 
            
            if self.depth != 50:
                bops_doublecfg[count-1][count1] += inplanes * 3 * 3 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits) #conv1
                bops_doublecfg[count1][count-1] += inplanes * 3 * 3 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)
                bops_singlecfg[count-1] += 5 * first_planes[0] * size * size * (args.quant_a_bits) 

                bops_doublecfg[count-1][count] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)  #conv2
                bops_doublecfg[count][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)  
                bops_singlecfg[count] += 5 * first_planes[1] * size * size * (args.quant_a_bits) 
            else:
                size *= 2
                bops_doublecfg[count-2][count1] += inplanes * 1 * 1 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)   #conv1
                bops_doublecfg[count1][count-2] += inplanes * 1 * 1 * first_planes[0] * size * size * (first_bits[0] * args.quant_a_bits)   
                bops_singlecfg[count-2] += 5 * first_planes[0] * size * size * (args.quant_a_bits) 

                size /= 2
                bops_doublecfg[count-2][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits)   #conv2
                bops_doublecfg[count-1][count-2] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (first_bits[1] * args.quant_a_bits) 
                bops_singlecfg[count-1] += 5 * first_planes[1] * size * size * (args.quant_a_bits) 

                bops_doublecfg[count][count-1] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (first_bits[2] * args.quant_a_bits)    #conv3
                bops_doublecfg[count-1][count] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (first_bits[2] * args.quant_a_bits) 
                bops_singlecfg[count] += 5 * first_planes[2] * size * size * (args.quant_a_bits) 

            for j in range(1, self.blocks[i]): #stage内后几个block
                inplanes = first_planes[-1]
                inbits = first_bits[-1]
                count1 = count
                if self.depth != 50:
                    bops_doublecfg[count1][count+j] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (later_bits[j-1] * args.quant_a_bits)
                    bops_doublecfg[count+j][count1] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (later_bits[j-1] * args.quant_a_bits)
                    bops_singlecfg[count+j] += 5 * later_planes[j-1] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count1][count+j] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_doublecfg[count+j][count1] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (args.quant_a_bits) 
                else:
                    bops_doublecfg[count1][count+1+2*(j-1)] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (later_bits[2*(j-1)] * args.quant_a_bits)
                    bops_doublecfg[count+1+2*(j-1)][count1] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (later_bits[2*(j-1)] * args.quant_a_bits)
                    bops_singlecfg[count+1+2*(j-1)] += 5 * later_planes[2*(j-1)] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count+2+2*(j-1)][count+1+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (later_bits[2*(j-1)+1] * args.quant_a_bits)
                    bops_doublecfg[count+1+2*(j-1)][count+2+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (later_bits[2*(j-1)+1] * args.quant_a_bits)
                    bops_singlecfg[count+2+2*(j-1)] += 5 * later_planes[2*(j-1)+1] * size * size * (args.quant_a_bits) 

                    bops_doublecfg[count+2+2*(j-1)][count1] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_doublecfg[count1][count+2+2*(j-1)] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (inbits * args.quant_a_bits)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (args.quant_a_bits) 
        bops_singlecfg[-1] += (2 * cfg[-1] + 1) * self.num_classes * args.a_bits * inbits
        return bops_singlecfg, bops_doublecfg, bops_squarecfg
    
    def cfg2fpbops(self, cfg, length, args):  # to simplify, only count convolution bops
        #通过cfg，计算每层(权重和激活值都不量化)的bops，length是cfg的长度
        blocks = self.blocks
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = 224
        size /= 2 # first conv layer s=2
        bops_singlecfg[0] += (3 * 7 * 7 * cfg[0] * size * size * (32 * 32) + 5 * cfg[0] * size * size * 32) # first layer, conv+bn+relu
        inplanes = cfg[0]
        size /= 2 # pooling s=2
        bops_singlecfg[0] += (3 * 3 * cfg[0] * size * size * 32) # maxpooling
        if self.depth != 50:
            count = 2
        else:
            count = 3
        count1 = 0
        if self.depth != 50:
            sub_cfgs = [cfg[1 : blocks[0]+2], #stage1
                        cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], #stage2
                        cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], #stage3
                        cfg[blocks[0]+blocks[1]+blocks[2]+4: ]] #stage4
        else:
            sub_cfgs = [cfg[1 : 2*blocks[0]+2],
                        cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1],
                        cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4],
                        cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ]]
        for i in range(4): # each layer
            planes = sub_cfgs[i]
            if self.depth != 50:
                first_planes = planes[0:2] #第一个block
                later_planes = planes[2:2+blocks[i]-1] #stage内后面所有的block
            else:
                first_planes = planes[0:3] #第一个block
                later_planes = planes[3:3+2*(blocks[i]-1)] #stage内后面所有的block
            if i in [1, 2, 3]: #第一个stage不做downsample
                size /= 2
                if self.depth != 50:
                    count += blocks[i-1]+1 #这里count的位置在下一个stage第一个block的最后一个conv
                else:
                    count += 2 * blocks[i-1] + 1
            bops_doublecfg[count][count1] += inplanes * 1 * 1 * first_planes[-1] * size * size * (32 * 32) # downsample layer
            bops_doublecfg[count1][count] += inplanes * 1 * 1 * first_planes[-1] * size * size * (32 * 32) 
            bops_singlecfg[count] += 5 * first_planes[-1] * size * size * (32) 
            
            if self.depth != 50:
                bops_doublecfg[count-1][count1] += inplanes * 3 * 3 * first_planes[0] * size * size * (32 * 32) #conv1
                bops_doublecfg[count1][count-1] += inplanes * 3 * 3 * first_planes[0] * size * size * (32 * 32)
                bops_singlecfg[count-1] += 5 * first_planes[0] * size * size * (32) 

                bops_doublecfg[count-1][count] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (32 * 32)  #conv2
                bops_doublecfg[count][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (32 * 32)  
                bops_singlecfg[count] += 5 * first_planes[1] * size * size * (32) 
            else:
                size *= 2
                bops_doublecfg[count-2][count1] += inplanes * 1 * 1 * first_planes[0] * size * size * (32 * 32)   #conv1
                bops_doublecfg[count1][count-2] += inplanes * 1 * 1 * first_planes[0] * size * size * (32 * 32)   
                bops_singlecfg[count-2] += 5 * first_planes[0] * size * size * (32) 

                size /= 2
                bops_doublecfg[count-2][count-1] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (32 * 32)   #conv2
                bops_doublecfg[count-1][count-2] += first_planes[0] * 3 * 3 * first_planes[1] * size * size * (32 * 32) 
                bops_singlecfg[count-1] += 5 * first_planes[1] * size * size * (32) 

                bops_doublecfg[count][count-1] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (32 * 32)    #conv3
                bops_doublecfg[count-1][count] += first_planes[1] * 1 * 1 * first_planes[2] * size * size * (32 * 32) 
                bops_singlecfg[count] += 5 * first_planes[2] * size * size * (32) 

            for j in range(1, self.blocks[i]): #stage内后几个block
                inplanes = first_planes[-1]
                count1 = count
                if self.depth != 50:
                    bops_doublecfg[count1][count+j] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (32 * 32)
                    bops_doublecfg[count+j][count1] += inplanes * 3 * 3 * later_planes[j-1] * size * size * (32 * 32)
                    bops_singlecfg[count+j] += 5 * later_planes[j-1] * size * size * (32) 

                    bops_doublecfg[count1][count+j] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (32 * 32)
                    bops_doublecfg[count+j][count1] += later_planes[j-1] * 3 * 3 * inplanes * size * size * (32 * 32)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (32) 
                else:
                    bops_doublecfg[count1][count+1+2*(j-1)] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (32 * 32)
                    bops_doublecfg[count+1+2*(j-1)][count1] += inplanes * 1 * 1 * later_planes[2*(j-1)] * size * size * (32 * 32)
                    bops_singlecfg[count+1+2*(j-1)] += 5 * later_planes[2*(j-1)] * size * size * (32) 

                    bops_doublecfg[count+2+2*(j-1)][count+1+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (32 * 32)
                    bops_doublecfg[count+1+2*(j-1)][count+2+2*(j-1)] += later_planes[2*(j-1)] * 3 * 3 * later_planes[2*(j-1)+1] * size * size * (32 * 32)
                    bops_singlecfg[count+2+2*(j-1)] += 5 * later_planes[2*(j-1)+1] * size * size * (32) 

                    bops_doublecfg[count+2+2*(j-1)][count1] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (32 * 32)
                    bops_doublecfg[count1][count+2+2*(j-1)] += later_planes[2*(j-1)+1] * 1 * 1 * inplanes * size * size * (32 * 32)
                    bops_singlecfg[count1] += 5 * inplanes * size * size * (32) 
        bops_singlecfg[-1] += (2 * cfg[-1] + 1) * self.num_classes * 32 * 32
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def feature_extract(self, x):
        tensor = []
        count = 0
        x = self.conv1(x)
        tensor.append(x)
        x = self.maxpool(x)
        if self.depth !=50: #如果不是resnet50，每个stage的第一个block，可以压缩2个conv，其他的block，只压缩第一个conv
            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in _layer:
                    if count == 0:
                        tensor.append(block.conv1(x))
                    x = block(x)
                    tensor.append(x) #这里代码应该写错了
                    count += 1
                count = 0
        else: #如果是resnet50，每个stage的第一个block，可以压缩3个conv，其他的block，只压缩前两个conv
            for _layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                count = 0
                for block in _layer:
                    if count == 0:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                        tensor.append(x)
                    else:
                        tensor.append(block.conv1(x))
                        tensor.append(block.conv2(block.conv1(x)))
                        x = block(x)
                    count += 1
        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'depth': self.depth,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base[self.depth],
            'dataset': 'ImageNet',
        }

