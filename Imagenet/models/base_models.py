import math
import torch
import torch.nn as nn
from utils import count_parameters

class MyNetwork(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def feature_extract(self, x):
        raise NotImplementedError

    @property
    def config(self):  # should include name/cfg/cfg_base/dataset
        raise NotImplementedError

    def cfg2params(self, cfg):
        raise NotImplementedError

    def cfg2flops(self, cfg):
        raise NotImplementedError

    def set_bn_param(self, momentum, eps): #设置bn的momentum和eps
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def get_bn_param(self): #获取bn层的momentum和eps
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    def init_model(self, model_init, init_div_groups=False): #初始化模型，包括conv2d，bn2d，linear，bn1d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif model_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data)
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_parameters(self, keys=None, mode='include'): #根据名称返回网络中的parameter
        if keys is None:
            for name, param in self.named_parameters():
                yield param
        elif mode == 'include': #若mode为'include'，就返回给定的parameter
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    yield param
        elif mode == 'exclude': #若mode为'exclude'，就不返回给定的parameter
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag:
                    yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def weight_parameters(self): #返回网络中所有的parameter
        return self.get_parameters()
