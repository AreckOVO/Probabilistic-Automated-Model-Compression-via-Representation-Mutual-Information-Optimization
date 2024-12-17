import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import pdb


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, res, shortcut):
        output = res + shortcut
        return output


# ********************* observers(统计min/max) *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':     # layer级(activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':   # channel级(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.q_level == 'FC':  # channel级(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]

        self.update_range(min_val, max_val)

class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:#没有值的时候，不比较大小，直接替换
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:#比较大小
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.out_channels = out_channels
        if self.q_level == 'L':
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.q_level == 'C':
            self.register_buffer('min_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.q_level == 'FC':
            self.register_buffer('min_val', torch.zeros((out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((out_channels, 1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.q_level == 'C':
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:#不比较大小，直接替换
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:#滑动平均
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

# ********************* quantizers（量化器，量化） *********************
# 取整(饱和/截断ste)
class Round(Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # 对称
        if q_type == 0:
            max_val = torch.max(torch.abs(observer_min_val), torch.abs(observer_max_val))
            min_val = -max_val
        # 非对称
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val= self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None

class Quantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag, union=False):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.activation_weight_flag = activation_weight_flag
        self.union = union
        self.q_type = 0
        # scale/zero_point/eps
        if self.observer.q_level == 'L':
            self.register_buffer('scale', torch.ones((1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((1), dtype=torch.float32))
        elif self.observer.q_level == 'C':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.q_level == 'FC':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.register_buffer('eps', torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))


    def update_qparams(self):
        raise NotImplementedError
    
    # 取整(ste)
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output
    
    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:#不支持二值化，二值化的实现在wbwtab里
            print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            if self.training:
                if not self.union:
                    self.observer(input)   # update observer_min and observer_max
                self.update_qparams()      # update scale and zero_point
            output = (torch.clamp(self.round(input / self.scale.clone() - self.zero_point,
                      self.observer.min_val / self.scale - self.zero_point,
                      self.observer.max_val / self.scale - self.zero_point, self.q_type),
                      self.quant_min_val, self.quant_max_val) + self.zero_point) * self.scale.clone()
        return output

    def extra_repr(self) -> str:
        return 'bits={}'.format(
            self.bits
        )

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:   # weight
            self.register_buffer('quant_min_val', torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        elif self.activation_weight_flag == 1: # activation
            self.register_buffer('quant_min_val', torch.tensor((-(1 << (self.bits - 1))), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')

    def update_bit(self):
        if self.activation_weight_flag == 0:   # weight
            self.quant_min_val = torch.tensor(-((1 << (self.bits - 1)) - 1))
            self.quant_max_val = torch.tensor((1 << (self.bits - 1)) - 1)
        elif self.activation_weight_flag == 1: # activation
            self.quant_min_val = torch.tensor((-(1 << (self.bits - 1))))
            self.quant_max_val = torch.tensor(((1 << (self.bits - 1)) - 1))
        else:
            print('activation_weight_flag error')
    
    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 32, 'bitwidth not supported'
        self.bits = refactored_bit
        if self.activation_weight_flag == 0:   # weight
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val = torch.tensor((1 << self.bits) - 2)
        elif self.activation_weight_flag == 1: # activation
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val =  torch.tensor((1 << self.bits) - 1)

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        if self.activation_weight_flag == 0:   # weight
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 2), dtype=torch.float32))
        elif self.activation_weight_flag == 1: # activation
            self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
            self.register_buffer('quant_max_val', torch.tensor(((1 << self.bits) - 1), dtype=torch.float32))
        else:
            print('activation_weight_flag error')
    
    def update_bit(self):
        if self.activation_weight_flag == 0:   # weight
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val = torch.tensor((1 << self.bits) - 2)
        elif self.activation_weight_flag == 1: # activation
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val =  torch.tensor((1 << self.bits) - 1)
        else:
            print('activation_weight_flag error')
    
    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 32, 'bitwidth not supported'
        self.bits = refactored_bit
        if self.activation_weight_flag == 0:   # weight
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val = torch.tensor((1 << self.bits) - 2)
        elif self.activation_weight_flag == 1: # activation
            self.quant_min_val = torch.tensor(0)
            self.quant_max_val = torch.tensor((1 << self.bits) - 1)

# 对称量化
class SymmetricQuantizer(SignedQuantizer):#对称量化继承有符号量化的
    def update_qparams(self):
        self.q_type = 0
        quant_range = float(self.quant_max_val - self.quant_min_val) / 2                                # quantized_range
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))     # float_range
        scale = float_range / quant_range                                                               # scale
        scale = torch.max(scale, self.eps)                                                              # processing for very small scale
        zero_point = torch.zeros_like(scale)                                                            # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):#非对称量化继承无符号量化
    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)                     # quantized_range
        float_range = self.observer.max_val - self.observer.min_val                      # float_range
        scale = float_range / quant_range                                                # scale
        scale = torch.max(scale, self.eps)                                               # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = sign * torch.floor(torch.abs(self.observer.min_val / scale) + 0.5)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# ********************* 量化卷积（同时量化A/W，并做卷积） *********************
class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=0,
                 weight_observer=0):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        
        self.q_type = q_type
        self.q_level = q_level
        self.weight_observer = weight_observer 
        
        if q_type == 0:#对称量化
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(#对于激活值，都使用滑动平均观察min和max，并使用layer-wise量化
                                                        q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:#对权重，可以直接观察min和max
                if q_level == 0:#channel-wise量化
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                            q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:#layer-wise量化
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=0)
            else:#对权重，可以使用滑动平均观察min和max
                if q_level == 0:#channel-wise量化
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:#layer-wise量化
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=0)
        else:#非对称量化
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(#对于激活值，都使用滑动平均观察min和max，并使用layer-wise量化
                                                            q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:#对权重，可以直接观察min和max
                if q_level == 0:#channel-wise量化
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:#layer-wise量化
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            else:#对权重，可以使用滑动平均观察min和max
                if q_level == 0:#channel-wise量化
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:#layer-wise量化
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
    
        

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if self.weight_quantizer.bits == 32:
            quant_weight = self.weight
        else:
            quant_weight = self.weight_quantizer(self.weight)
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output



def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)
def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)
def reshape_to_bias(input):
    return input.reshape(-1)

# ********************* bn融合_量化卷积（bn融合后，同时量化A/W，并做卷积） *********************
class QuantBNFuseConv2d(QuantConv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 eps=1e-5,
                 momentum=0.1,
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=0,
                 weight_observer=0,
                 pretrained_model=False,
                 bn_fuse_calib=False):
        super(QuantBNFuseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                                bias, padding_mode)
        self.num_flag = 0
        self.pretrained_model = pretrained_model
        self.bn_fuse_calib = bn_fuse_calib
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros((out_channels), dtype=torch.float32))
        self.register_buffer('running_var', torch.ones((out_channels), dtype=torch.float32))
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='C', out_channels=out_channels), activation_weight_flag=0)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            
            self.w_bits = self.weight_quantizer.bits

    def forward(self, input):        
            
        # 训练态
        if self.training:
            # 先做普通卷积得到A，以取得BN参数
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                            self.groups)
            # 更新BN统计参数（batch和running）
            dims = [dim for dim in range(4) if dim != 1]#dims=[0,2,3]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if not self.pretrained_model:#如果没有预训练模型，可能没有running_mean和running_var
                    if self.num_flag == 0:
                        self.num_flag += 1
                        running_mean = batch_mean
                        running_var = batch_var
                    else:
                        running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                        running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                else:
                    running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                    running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
            # bn融合
            if self.bias is not None:
                bias_fused = reshape_to_bias(self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
                bias_fused = reshape_to_bias(self.beta - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps)))  # b融batch
                # bn融合不校准
            if not self.bn_fuse_calib:    
                weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(batch_var + self.eps))           # w融batch
                # bn融合校准
            else:
                weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))    # w融running
        # 测试态
        else:
            if self.bias is not None:
                bias_fused = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias_fused = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
            weight_fused = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))                      # w融running
        
        # 量化A和bn融合后的W
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(weight_fused)    
           
        # 量化卷积
        if self.training:  # 训练态
            # bn融合不校准
            if not self.bn_fuse_calib:
                output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                                    self.groups)
            # bn融合校准
            else:
                output = F.conv2d(quant_input, quant_weight, None, self.stride, self.padding, self.dilation,
                                    self.groups)  # 注意，这里不加bias（self.bias为None）
                # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
                output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
                output += reshape_to_activation(bias_fused)
        else:  # 测试态
            output = F.conv2d(quant_input, quant_weight, bias_fused, self.stride, self.padding, self.dilation,
                                self.groups)  # 注意，这里加bias，做完整的conv+bn
      
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 q_type=0,
                 q_level=0,
                 weight_observer=0):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.q_type = q_type
        self.q_level = q_level
        self.weight_observer = weight_observer

        if q_type == 0:
            self.activation_quantizer = SymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='FC', out_channels=out_features), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='FC', out_channels=out_features), activation_weight_flag=0)
                else:
                    self.weight_quantizer = SymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
        else:
            self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, observer=MovingAverageMinMaxObserver(
                                                            q_level='L', out_channels=None), activation_weight_flag=1)
            if weight_observer == 0:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='FC', out_channels=out_features), activation_weight_flag=0)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
            else:
                if q_level == 0:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='FC', out_channels=out_features), activation_weight_flag=0)
                else:
                    self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, observer=MovingAverageMinMaxObserver(
                                                                q_level='L', out_channels=None), activation_weight_flag=0)
    

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if self.weight_quantizer.bits == 32:
            quant_weight = self.weight
        else:
            quant_weight = self.weight_quantizer(self.weight)        
        output = F.linear(quant_input, quant_weight, self.bias)
        return output

def quantize_conv(conv, w_bits, args):
    if not isinstance(conv, QuantConv2d):
        tmp = conv
        conv = QuantConv2d(in_channels = conv.in_channels, out_channels = conv.out_channels,
                                            kernel_size = conv.kernel_size, stride = conv.stride,
                                            padding = conv.padding, dilation = conv.dilation,
                                            groups=conv.groups, bias=conv.bias is not None, 
                                            padding_mode=conv.padding_mode,
                                            a_bits=args.quant_a_bits, w_bits=w_bits, 
                                            q_type=args.q_type, q_level=args.q_level, 
                                            weight_observer=args.weight_observer)
        if tmp.bias is not None:
            conv.bias.data = tmp.bias                    
        conv.weight.data = tmp.weight
    elif w_bits != conv.weight_quantizer.bits:
        conv.weight_quantizer.bitwidth_refactor(w_bits)
        # conv.activation_quantizer.bitwidth_refactor(args.quant_a_bits)
    return conv

def quantize_fc(fc, w_bits, args):
    if not isinstance(fc, QuantLinear):
        tmp = fc
        fc = QuantLinear(in_features=fc.in_features, out_features=fc.out_features,
                                        bias=fc.bias is not None, 
                                        a_bits=32, w_bits=w_bits,
                                        q_type=args.q_type, q_level=args.q_level,
                                        weight_observer=args.weight_observer)
            
        if tmp.bias is not None:
            fc.bias.data = tmp.bias
        fc.weight.data = tmp.weight
    elif w_bits != fc.weight_quantizer.bits:
        fc.weight_quantizer.bitwidth_refactor(w_bits)
    return fc