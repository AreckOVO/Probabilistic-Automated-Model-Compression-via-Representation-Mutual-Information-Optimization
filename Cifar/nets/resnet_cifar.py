from utils import *
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from nets.base_models import *
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from utils import quantize
from utils.quantize import *

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1 #没用到
    def __init__(self, in_planes, planes, stride=1, affine=True):
        super(BasicBlock, self).__init__()
        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes, affine=affine)
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],  #这种做法planes一定要比in_planes大，否则会报错
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            # x[:, :, ::2, ::2] 表示对 x 的第 2 和第 3 个维度进行 stride = 2 的 slice. 
            # 经过这一步操作后，输入 F.pad 的 tensor 的长和宽已经变成了以前的 1/2.
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(MyNetwork):
    def __init__(self, depth=20, num_classes=10, cfg=None, cutout=False, quant_cfg=None): #cutout应该是用来数据增强的
        super(ResNet_CIFAR, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        cfg_base = []
        n = (depth-2)//6 #n是每个stage的block数
        for i in [16, 32, 64]:
            for j in range(n):
                cfg_base.append(i)

        if cfg is None:
            cfg = cfg_base
        num_blocks = []
        if depth==20:
            num_blocks = [3, 3, 3]
        elif depth==32:
            num_blocks = [5, 5, 5]
        elif depth==44:
            num_blocks = [7, 7, 7]
        elif depth==56:
            num_blocks = [9, 9, 9]
        elif depth==110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.cfg_base = cfg_base
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_planes = 16
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) #第一层不剪枝
        bn1 = nn.BatchNorm2d(16)
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.layer1 = self._make_layer(block, cfg[0:n], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[n:2*n], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2*n:], num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) #[2,1,1]
        layers = []
        for i in range(len(strides)):
            layers.append(('block_%d'%i, block(self.in_planes, planes[i], strides[i])))
            self.in_planes = planes[i]
        return nn.Sequential(OrderedDict(layers))

    def quantize_net(self, quant_cfg, args):
        # self.conv_bn.conv = quantize_conv(self.conv_bn.conv, 8, args)#第一层量化为8bit
        # self.conv_bn.conv.weight_quantizer.bitwidth_refactor(refactored_bit = 8)
        # self.conv_bn.conv.activation_quantizer.bitwidth_refactor(refactored_bit = 8)
        for i in range(len(quant_cfg)):
            self.quantize_layer(quant_cfg[i], i, args)
        self.linear = quantize_fc(self.linear, 32, args)
        self.linear.weight_quantizer.bitwidth_refactor(refactored_bit = 32)
        self.linear.activation_quantizer.bitwidth_refactor(refactored_bit = 32)
        

    def quantize_layer(self, w_bits, idx, args):
        num_blocks = self.num_blocks
        if idx in range(0, num_blocks[0]):
            tmp_block = self.layer1[idx - 0]
            tmp_block.conv_bn1.conv = quantize_conv(tmp_block.conv_bn1.conv, w_bits, args)
            tmp_block.conv_bn2.conv = quantize_conv(tmp_block.conv_bn2.conv, w_bits, args)
        elif idx in range(num_blocks[0], num_blocks[0]+num_blocks[1]):
            tmp_block = self.layer2[idx - num_blocks[0]]
            tmp_block.conv_bn1.conv = quantize_conv(tmp_block.conv_bn1.conv, w_bits, args)
            tmp_block.conv_bn2.conv = quantize_conv(tmp_block.conv_bn2.conv, w_bits, args)
        elif idx in range(num_blocks[0]+num_blocks[1], num_blocks[0]+num_blocks[1]+num_blocks[2]):
            tmp_block = self.layer3[idx - (num_blocks[0]+num_blocks[1])]
            tmp_block.conv_bn1.conv = quantize_conv(tmp_block.conv_bn1.conv, w_bits, args)
            tmp_block.conv_bn2.conv = quantize_conv(tmp_block.conv_bn2.conv, w_bits, args)
        

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def feature_extract(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        tensor = []
        out = F.relu(self.conv_bn(x))
        for i in [self.layer1, self.layer2, self.layer3]:
            for _layer in i:
                out = _layer(out)
                if type(_layer) is BasicBlock:
                    tensor.append(out)
        return tensor

    def cfg2params(self, cfg):
        params = 0
        params += (3 * 3 * 3 * 16 + 16 * 2) # conv1+bn1
        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                params += (in_c * 3 * 3 * c + 2 * c + c * 3 * 3 * c + 2 * c) # per block params
                if in_c != c:
                    params += in_c * c # shortcut
                in_c = c
                cfg_idx += 1
        params += (self.cfg[-1] + 1) * self.num_classes # fc layer
        return params

    def cfg2flops(self, cfg):  # to simplify, only count convolution flops
        size = 32
        flops = 0
        flops += (3 * 3 * 3 * 16 * 32 * 32 + 16 * 32 * 32 * 4) # conv1+bn1
        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                flops += (in_c * 3 * 3 * c * size * size + c * size * size * 4 + c * 3 * 3 * c * size * size + c * size * size * 4) # per block flops
                if in_c != c:
                    flops += in_c * c * size * size # shortcut
                in_c = c
                cfg_idx += 1
        flops += (2 * self.cfg[-1] + 1) * self.num_classes # fc layer
        return flops

    def cfg2flops_perlayer(self, cfg, length):  # to simplify, only count convolution flops
        size = 32
        flops_singlecfg = [0 for j in range(length)]
        flops_doublecfg = np.zeros((length, length))
        flops_squarecfg = [0 for j in range(length)]

        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                if i==0 and j==0:
                    flops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4 + in_c * 3 * 3 * c * size * size)#stage1 block1的conv1、bn1、bn2 
                    flops_squarecfg[cfg_idx] += c * 3 * 3 * c * size * size #stage1 block1的conv2
                else:
                    flops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4)#bn1+bn2
                    flops_doublecfg[cfg_idx-1][cfg_idx] += in_c * 3 * 3 * c * size * size #conv1
                    flops_doublecfg[cfg_idx][cfg_idx-1] += in_c * 3 * 3 * c * size * size
                    flops_squarecfg[cfg_idx] += (c * 3 * 3 * c * size * size ) #conv2
                if in_c != c:
                    flops_doublecfg[cfg_idx][cfg_idx-1] += in_c * c * size * size # shortcut
                    flops_doublecfg[cfg_idx-1][cfg_idx] += in_c * c * size * size
                in_c = c
                cfg_idx += 1

        flops_singlecfg[-1] += 2 * self.cfg[-1] * self.num_classes # fc layer
        return flops_singlecfg, flops_doublecfg, flops_squarecfg

    """"CAL BITOPS"""
    """"CAL BITOPS"""
    def cfg2bops(self, cfg, quant_cfg, length, args):  # to simplify, only count convolution bops
        size = 32
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]

        in_c = 16
        in_bit = 8
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                bit = quant_cfg[cfg_idx]
                if i==0 and j==0:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 * args.quant_a_bits + c * size * size * 4 * args.quant_a_bits + in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits))#stage1 block1的bn1、bn2、conv1 
                    bops_squarecfg[cfg_idx] += c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)#stage1 block1的conv2
                else:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4) * args.quant_a_bits#bn1+bn2
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)#conv1
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)
                    bops_squarecfg[cfg_idx] += (c * 3 * 3 * c * size * size ) * (bit * args.quant_a_bits)#conv2
                if in_c != c:
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * c * size * size * (bit * args.quant_a_bits)# shortcut pad
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * c * size * size * (bit * args.quant_a_bits)
                in_c = c
                in_bit = bit
                cfg_idx += 1

        bops_singlecfg[-1] += 2 * self.cfg[-1] * self.num_classes * 8 * 8 # fc layer
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
        size = 32
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]

        in_c = 16
        in_bit = 8
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                bit = quant_cfg[cfg_idx]
                if i==0 and j==0:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 * args.quant_a_bits + c * size * size * 4 * args.quant_a_bits + in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits))#stage1 block1的bn1、bn2、conv1 
                    bops_squarecfg[cfg_idx] += c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)#stage1 block1的conv2
                else:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4) * args.quant_a_bits#bn1+bn2
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)#conv1
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * 3 * 3 * c * size * size * (bit * args.quant_a_bits)
                    bops_squarecfg[cfg_idx] += (c * 3 * 3 * c * size * size ) * (bit * args.quant_a_bits)#conv2
                if in_c != c:
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * c * size * size * (bit * args.quant_a_bits)# shortcut pad
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * c * size * size * (bit * args.quant_a_bits)
                in_c = c
                in_bit = bit
                cfg_idx += 1

        bops_singlecfg[-1] += 2 * self.cfg[-1] * self.num_classes * 8 * 8 # fc layer
        return bops_singlecfg, bops_doublecfg, bops_squarecfg

    def cfg2fpbops(self, cfg, length, args):  # to simplify, only count convolution bops
        #权重 激活值都不量化的bops
        size = 32
        bops_singlecfg = [0 for j in range(length)]
        bops_doublecfg = np.zeros((length, length))
        bops_squarecfg = [0 for j in range(length)]

        in_c = 16
        cfg_idx = 0
        for i in range(3):
            num_blocks = self.num_blocks[i]
            if i==1 or i==2:
                size = size // 2
            for j in range(num_blocks):
                c = cfg[cfg_idx]
                if i==0 and j==0:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 * 32 + c * size * size * 4 * 32 + in_c * 3 * 3 * c * size * size * (32 * 32))#stage1 block1的conv1、bn1、bn2 
                    bops_squarecfg[cfg_idx] += c * 3 * 3 * c * size * size * (32 * 32)#stage1 block1的conv2
                else:
                    bops_singlecfg[cfg_idx] += (c * size * size * 4 + c * size * size * 4) * 32#bn1+bn2
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * 3 * 3 * c * size * size * (32 * 32)#conv1
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * 3 * 3 * c * size * size * (32 * 32)
                    bops_squarecfg[cfg_idx] += (c * 3 * 3 * c * size * size ) * (32 * 32)#conv2
                if in_c != c:
                    bops_doublecfg[cfg_idx][cfg_idx-1] += in_c * c * size * size * 32 * 32 # shortcut pad
                    bops_doublecfg[cfg_idx-1][cfg_idx] += in_c * c * size * size * 32 * 32
                in_c = c
                cfg_idx += 1

        bops_singlecfg[-1] += 2 * self.cfg[-1] * self.num_classes * 32 * 32 # fc layer
        bops = []
        for i in range(length):
            bops.append(bops_singlecfg[i])
            bops.append(bops_squarecfg[i])
        for i in range(1,length):
            for j in range(i):
                bops.append(bops_doublecfg[i][j])
        sum_bops = np.sum(bops)
        return sum_bops


    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': self.cfg_base,
            'dataset': 'cifar10',
        }

