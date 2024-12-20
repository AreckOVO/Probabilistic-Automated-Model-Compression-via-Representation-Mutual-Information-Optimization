import torch.nn as nn
from models.base_models import *
from collections import OrderedDict
import sys
import numpy as np
from utils import quantize
from utils.quantize import *

def conv_bn(inp, oup, stride):
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=False)), #kernel_size, stride, padding
                                      ('bn', nn.BatchNorm2d(oup)),
                                      ('relu', nn.ReLU6(inplace=True))]))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
                                      ('bn', nn.BatchNorm2d(oup)),
                                      ('relu', nn.ReLU6(inplace=True))]))

def conv_dw(inp, oup, stride):
    conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)), #dw
                                       ('bn', nn.BatchNorm2d(inp)),
                                       ('relu', nn.ReLU(inplace=True))]))
    conv2 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)), #pw
                                       ('bn', nn.BatchNorm2d(oup)),
                                       ('relu', nn.ReLU(inplace=True))]))
    return nn.Sequential(conv1, conv2)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup # self.stride == 1且 inp == oup时use_res_connect为True
        #dw不改变通道数，pw可能会改变通道数（取决于expand_ratio）
        if expand_ratio == 1:
            dw = nn.Sequential(OrderedDict([('conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                                            ('bn', nn.BatchNorm2d(hidden_dim)),
                                            ('relu', nn.ReLU6(inplace=True))]))
            pw = nn.Sequential(OrderedDict([('conv', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                                            ('bn', nn.BatchNorm2d(oup))]))
            self.conv = nn.Sequential(dw, pw)
        else:
            pw = nn.Sequential(OrderedDict([('conv', nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                                            ('bn', nn.BatchNorm2d(hidden_dim)),
                                            ('relu', nn.ReLU6(inplace=True))]))
            dw = nn.Sequential(OrderedDict([('conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                                            ('bn', nn.BatchNorm2d(hidden_dim)),
                                            ('relu', nn.ReLU6(inplace=True))]))
            pwl = nn.Sequential(OrderedDict([('conv', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                                             ('bn', nn.BatchNorm2d(oup))]))
            self.conv = nn.Sequential(pw, dw, pwl)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(MyNetwork):
    def __init__(self, cfg=None, num_classes=1000, dropout=0.2):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        if cfg==None:
            cfg = [32, 16, 24, 32, 64, 96, 160, 320]
        input_channel = cfg[0]
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]

        # building first layer
        # input_channel = int(input_channel * width_mult)
        self.cfg = cfg
        self.cfgs_base = [32, 16, 24, 32, 64, 96, 160, 320]
        self.dropout = dropout
        self.last_channel = 1280
        self.num_classes = num_classes
        self.features = [conv_bn(3, input_channel, 2)] #inp, oup, stride,第一层的conv
        self.interverted_residual_setting = interverted_residual_setting
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting: #expand_ratio, output_channel, num, stride
            output_channel = c
            for i in range(n):
                if i == 0: #每个stage内第一个block的stride为s
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else: #其他block的stride为1
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel #下一个block的input_channel等于上一个block的output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.last_channel, num_classes),
        )
    
    def quantize_net(self, quant_cfg, args):
        for i in range(len(quant_cfg)):
            self.quantize_layer(quant_cfg[i], i, args)
        self.features[18].conv = quantize_conv(self.features[18].conv, 32, args)
        self.features[18].conv.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
        self.features[18].conv.activation_quantizer.bitwidth_refactor(refactored_bit = 32)

        self.classifier[0] = quantize_fc(self.classifier[0], 32, args)
        self.classifier[0].weight_quantizer.bitwidth_refactor(refactored_bit = 32)
        self.classifier[0].activation_quantizer.bitwidth_refactor(refactored_bit = 32)
    
    def quantize_layer(self, w_bits, idx, args):
        if idx == 0:           
            self.features[0].conv = quantize_conv(self.features[0].conv, w_bits, args)#第一层量化为32bit
            self.features[0].conv.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
            self.features[0].conv.activation_quantizer.bitwidth_refactor(refactored_bit = 32)            
            return
        if idx == 1:
            self.features[1].conv[0].conv = quantize_conv(self.features[1].conv[0].conv, w_bits, args)
            self.features[1].conv[1].conv = quantize_conv(self.features[1].conv[1].conv, w_bits, args)
        elif idx == 2:
            for k in [2,3]:
                self.features[k].conv[0].conv = quantize_conv(self.features[k].conv[0].conv, w_bits, args)
                self.features[k].conv[1].conv = quantize_conv(self.features[k].conv[1].conv, w_bits, args)
                self.features[k].conv[2].conv = quantize_conv(self.features[k].conv[2].conv, w_bits, args)
        elif idx == 3:
            for k in [4,5,6]:
                self.features[k].conv[0].conv = quantize_conv(self.features[k].conv[0].conv, w_bits, args)
                self.features[k].conv[1].conv = quantize_conv(self.features[k].conv[1].conv, w_bits, args)
                self.features[k].conv[2].conv = quantize_conv(self.features[k].conv[2].conv, w_bits, args)
        elif idx == 4:
            for k in [7,8,9,10]:
                self.features[k].conv[0].conv = quantize_conv(self.features[k].conv[0].conv, w_bits, args)
                self.features[k].conv[1].conv = quantize_conv(self.features[k].conv[1].conv, w_bits, args)
                self.features[k].conv[2].conv = quantize_conv(self.features[k].conv[2].conv, w_bits, args)
        elif idx == 5:
            for k in [11,12,13]:
                self.features[k].conv[0].conv = quantize_conv(self.features[k].conv[0].conv, w_bits, args)
                self.features[k].conv[1].conv = quantize_conv(self.features[k].conv[1].conv, w_bits, args)
                self.features[k].conv[2].conv = quantize_conv(self.features[k].conv[2].conv, w_bits, args)
        elif idx == 6:
            for k in [14,15,16]:
                self.features[k].conv[0].conv = quantize_conv(self.features[k].conv[0].conv, w_bits, args)
                self.features[k].conv[1].conv = quantize_conv(self.features[k].conv[1].conv, w_bits, args)
                self.features[k].conv[2].conv = quantize_conv(self.features[k].conv[2].conv, w_bits, args)
        elif idx == 7:
            self.features[17].conv[0].conv = quantize_conv(self.features[17].conv[0].conv, w_bits, args)
            self.features[17].conv[1].conv = quantize_conv(self.features[17].conv[1].conv, w_bits, args)
            self.features[17].conv[2].conv = quantize_conv(self.features[17].conv[2].conv, w_bits, args)



    def cfg2params(self, cfg):
        params = 0.
        params += (3 * 3 * 3 * cfg[0] + 2 * cfg[0]) # first layer
        input_channel = cfg[0]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1:
                        params += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params += (1 * 1 * hidden_dim * output_channel + 2 * output_channel)
                    else:
                        params += (1 * 1 * input_channel * hidden_dim + 2 * hidden_dim)
                        params += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params += (1 * 1 * hidden_dim * output_channel + 2 * output_channel)
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        params += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params += (1 * 1 * hidden_dim * output_channel + 2 * output_channel)
                    else:
                        params += (1 * 1 * input_channel * hidden_dim + 2 * hidden_dim)
                        params += (3 * 3 * hidden_dim + 2 * hidden_dim)
                        params += (1 * 1 * hidden_dim * output_channel + 2 * output_channel)
                input_channel = output_channel
        params += (1 * 1 * input_channel * self.last_channel + 2 * self.last_channel) # final 1x1 conv
        params += ((self.last_channel + 1) * self.num_classes) # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        flops = 0.
        size = size//2
        flops += (3 * 3 * 3 * cfg[0] + 0 * cfg[0]) * size * size # first layer
        input_channel = cfg[0]
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s==2:
                        size = size//2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1:
                        flops += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                    else:
                        size = size * s
                        flops += (1 * 1 * input_channel * hidden_dim + 0 * hidden_dim) * size * size
                        size = size // s
                        flops += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        flops += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                    else:
                        flops += (1 * 1 * input_channel * hidden_dim + 0 * hidden_dim) * size * size
                        flops += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                input_channel = output_channel
        flops += (1 * 1 * input_channel * self.last_channel + 0 * self.last_channel) * size * size # final 1x1 conv
        flops += ((2 * self.last_channel - 1) * self.num_classes) # fc layer
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]
        size = size//2
        flops_singlecfg[0] += (3 * 3 * 3 * cfg[0] + 0 * cfg[0]) * size * size # first layer
        input_channel = cfg[0]
        count = 0
        for t, c, n, s in interverted_residual_setting: #k是新增的计数器
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s==2:
                        size = size//2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1: 
                        flops_singlecfg[count] += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size #dw
                        flops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size #pw
                        flops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                    else:
                        size = size * s
                        flops_squarecfg[count] += (1 * 1 * input_channel * hidden_dim + 0 * hidden_dim) * size * size
                        size = size // s
                        flops_singlecfg[count] += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                        flops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        flops_singlecfg[count+1] += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                    else:
                        flops_squarecfg[count+1] += (1 * 1 * input_channel * hidden_dim + 0 * hidden_dim) * size * size
                        flops_singlecfg[count+1] += (3 * 3 * hidden_dim + 0 * hidden_dim) * size * size
                        flops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel + 0 * output_channel) * size * size
                input_channel = output_channel
            count += 1
        flops_singlecfg[count] += (1 * 1 * input_channel * self.last_channel + 0 * self.last_channel) * size * size # final 1x1 conv

        return flops_singlecfg, flops_doublecfg, flops_squarecfg
    
    """"CAL BITOPS"""
    """"CAL BITOPS"""
    def cfg2bops(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (quant_cfg[0] * args.quant_a_bits) + 0 * cfg[0]) * size * size # first layer
        input_channel = cfg[0]
        input_bit = quant_cfg[0]
        count = 0
        for k, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s==2:
                        size = size//2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                    else:
                        size = size * s
                        bops_squarecfg[count] += (1 * 1 * input_channel * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        size = size // s
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                    else:
                        bops_squarecfg[count+1] += (1 * 1 * input_channel * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                input_channel = output_channel
            count += 1
        bops_singlecfg[count] += (1 * 1 * input_channel * self.last_channel * (quant_cfg[-1] * args.quant_a_bits) + 0 * self.last_channel) * size * size # final 1x1 conv
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops

    def cfg2bops_perlayer(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (quant_cfg[0] * args.quant_a_bits) + 0 * cfg[0]) * size * size # first layer
        input_channel = cfg[0]
        input_bit = quant_cfg[0]
        count = 0
        for k, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s==2:
                        size = size//2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                    else:
                        size = size * s
                        bops_squarecfg[count] += (1 * 1 * input_channel * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        size = size // s
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                    else:
                        bops_squarecfg[count+1] += (1 * 1 * input_channel * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (quant_cfg[k+1] * args.quant_a_bits) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (quant_cfg[k+1] * args.quant_a_bits) + 0 * output_channel) * size * size
                input_channel = output_channel
            count += 1
        bops_singlecfg[count] += (1 * 1 * input_channel * self.last_channel * (quant_cfg[-1] * args.quant_a_bits) + 0 * self.last_channel) * size * size # final 1x1 conv

        return bops_singlecfg, bops_doublecfg, bops_squarecfg

    def cfg2fpbops(self, cfg, length, args):  # to simplify, only count convolution flops
        interverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1], 1, 1],
            [6, cfg[2], 2, 2],
            [6, cfg[3], 3, 2],
            [6, cfg[4], 4, 2],
            [6, cfg[5], 3, 1],
            [6, cfg[6], 3, 2],
            [6, cfg[7], 1, 1],
        ]
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (32 * 32) + 0 * cfg[0]) * size * size # first layer
        input_channel = cfg[0]
        count = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                hidden_dim = round(input_channel * t)
                if i == 0:
                    if s==2:
                        size = size//2
                    # self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                    else:
                        size = size * s
                        bops_squarecfg[count] += (1 * 1 * input_channel * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        size = size // s
                        bops_singlecfg[count] += (3 * 3 * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        bops_doublecfg[count][count+1] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                        bops_doublecfg[count+1][count] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                else:
                    # self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    if t==1:
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                    else:
                        bops_squarecfg[count+1] += (1 * 1 * input_channel * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        bops_singlecfg[count+1] += (3 * 3 * hidden_dim * (32 * 32) + 0 * hidden_dim) * size * size
                        bops_squarecfg[count+1] += (1 * 1 * hidden_dim * output_channel * (32 * 32) + 0 * output_channel) * size * size
                input_channel = output_channel
            count += 1
        bops_singlecfg[count] += (1 * 1 * input_channel * self.last_channel * (32 * 32) + 0 * self.last_channel) * size * size # final 1x1 conv
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops

    def get_flops(self, x):
        return self.cfg2flops(cfg=self.cfg)

    def get_params(self):
        return self.cfg2param(self.cfg)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def feature_extract(self, x):
        tensor = []
        count = 0
        for _layer in self.features:
            x = _layer(x)
            if type(_layer) is InvertedResidual:
                count += 1
            if count in [1,3,6,10,13,16,17]:
                tensor.append(x)
        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base,
            'dataset': 'ImageNet',
        }


class MobileNet(MyNetwork):
    def __init__(self, cfg=None, num_classes=1000):
        super(MobileNet, self).__init__()
        if cfg==None:
            cfg = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        self.cfg = cfg
        self.cfgs_base = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        self._cfg = [cfg[1], (cfg[2], 2), cfg[3], (cfg[4], 2), cfg[5], (cfg[6], 2),\
                     cfg[7], cfg[8], cfg[9], cfg[10], cfg[11], (cfg[12], 2), cfg[13]]
        in_planes = cfg[0]
        self.conv1 = conv_bn(3, in_planes, stride=2)
        self.num_classes = num_classes
        self.features = self._make_layers(in_planes, self._cfg, conv_dw)
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling
        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def quantize_net(self, quant_cfg, args):
        for i in range(len(quant_cfg)):
            self.quantize_layer(quant_cfg[i], i, args)        

        self.classifier[0] = quantize_fc(self.classifier[0], 32, args)
        self.classifier[0].weight_quantizer.bitwidth_refactor(refactored_bit = 32)
        self.classifier[0].activation_quantizer.bitwidth_refactor(refactored_bit = 32)
    
    def quantize_layer(self, w_bits, idx, args):
        if idx == 0:           
            self.conv1.conv = quantize_conv(self.conv1.conv, w_bits, args)#第一层量化为32bit
            self.conv1.conv.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
            self.conv1.conv.activation_quantizer.bitwidth_refactor(refactored_bit = 32)            
            return
        else:
            self.features[idx-1][0].conv = quantize_conv(self.features[idx-1][0].conv, w_bits, args)
            self.features[idx-1][1].conv = quantize_conv(self.features[idx-1][1].conv, w_bits, args)


    def cfg2param(self, cfg):
        params = 0.
        params += (3 * 3 * 3 * cfg[0] + 2 * cfg[0]) # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            out_c = cfg[i]
            params += (3 * 3 * in_c + 2 * in_c + 1 * 1 * in_c * out_c + 2 * out_c)
            in_c = out_c
        params += ((out_c + 1) * self.num_classes) # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        size = 224
        flops = 0.
        size = size//2
        flops += (3 * 3 * 3 * cfg[0] + 0 * cfg[0]) * size * size # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            if i in [2, 4, 6, 12]:
                size = size//2
            out_c = cfg[i]
            flops += (3 * 3 * in_c + 0 * in_c + 1 * 1 * in_c * out_c + 0 * out_c) * size * size
            in_c = out_c
        flops += ((2 * out_c - 1) * self.num_classes) # fc layer
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        size = 224
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]
        size = size//2
        flops_singlecfg[0] += (3 * 3 * 3 * cfg[0] + 0 * cfg[0]) * size * size # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            if i in [2, 4, 6, 12]:
                size = size//2
            out_c = cfg[i]
            flops_singlecfg[i-1] += 3 * 3 * in_c * size * size
            flops_doublecfg[i-1][i] += 1 * 1 * in_c * out_c * size * size
            flops_doublecfg[i][i-1] += 1 * 1 * in_c * out_c * size * size
            in_c = out_c
        flops_singlecfg[length-1] += 2 * out_c * self.num_classes # fc layer
        return flops_singlecfg, flops_doublecfg, flops_squarecfg
    
    """"CAL BITOPS"""
    """"CAL BITOPS"""
    def cfg2bops(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution flops
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (quant_cfg[0] * args.quant_a_bits) + 0 * cfg[0]) * size * size # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            if i in [2, 4, 6, 12]:
                size = size//2
            out_c = cfg[i]
            bops_singlecfg[i-1] += 3 * 3 * in_c * (quant_cfg[i-1] * args.quant_a_bits) * size * size  #dw
            bops_doublecfg[i-1][i] += 1 * 1 * in_c * out_c * (quant_cfg[i] * args.quant_a_bits) * size * size #pw
            bops_doublecfg[i][i-1] += 1 * 1 * in_c * out_c * (quant_cfg[i] * args.quant_a_bits) * size * size
            in_c = out_c
        bops_singlecfg[length-1] += 2 * out_c * (quant_cfg[-1] * args.quant_a_bits) * self.num_classes # fc layer
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops
    
    def cfg2bops_perlayer(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution flops
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (quant_cfg[0] * args.quant_a_bits) + 0 * cfg[0]) * size * size # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            if i in [2, 4, 6, 12]:
                size = size//2
            out_c = cfg[i]
            bops_singlecfg[i-1] += 3 * 3 * in_c * (quant_cfg[i-1] * args.quant_a_bits) * size * size
            bops_doublecfg[i-1][i] += 1 * 1 * in_c * out_c * (quant_cfg[i] * args.quant_a_bits) * size * size
            bops_doublecfg[i][i-1] += 1 * 1 * in_c * out_c * (quant_cfg[i] * args.quant_a_bits) * size * size
            in_c = out_c
        bops_singlecfg[length-1] += 2 * out_c * (quant_cfg[-1] * args.quant_a_bits) * self.num_classes # fc layer
        return bops_singlecfg, bops_doublecfg, bops_squarecfg
    
    def cfg2fpbops(self, cfg, length, args):  # to simplify, only count convolution flops
        size = 224
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]
        size = size//2
        bops_singlecfg[0] += (3 * 3 * 3 * cfg[0] * (32 * 32) + 0 * cfg[0]) * size * size # first layer
        in_c = cfg[0]
        for i in range(1, len(cfg)):
            if i in [2, 4, 6, 12]:
                size = size//2
            out_c = cfg[i]
            bops_singlecfg[i-1] += 3 * 3 * in_c * (32 * 32) * size * size
            bops_doublecfg[i-1][i] += 1 * 1 * in_c * out_c * (32 * 32) * size * size
            bops_doublecfg[i][i-1] += 1 * 1 * in_c * out_c * (32 * 32) * size * size
            in_c = out_c
        bops_singlecfg[length-1] += 2 * out_c * (32 * 32) * self.num_classes # fc layer
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops

    def feature_extract(self, x):
        tensor = []
        x = self.conv1(x)
        for _layer in self.features:
            x = _layer(x)
            if type(_layer) is nn.Sequential:
                tensor.append(x)
        x = x.mean(3).mean(2)  # global average pooling
        x = self.classifier(x)
        tensor.append(x)

        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base,
            'dataset': 'ImageNet',
        }