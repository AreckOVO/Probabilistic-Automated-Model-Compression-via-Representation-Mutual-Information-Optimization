import torch
import numpy as np
import torch.nn as nn
import sys

def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

def raw2cfg(model, raw_ratios, flops, p=False, div=8):
    left = 0
    right = 50
    scale = 0
    cfg = None
    current_flops = 0
    base_channels = model.config['cfg_base']
    cnt = 0
    while (True):
        cnt += 1
        scale = (left + right) / 2
        scaled_ratios = raw_ratios * scale
        for i in range(len(scaled_ratios)):
            scaled_ratios[i] = max(0.1, scaled_ratios[i])
            scaled_ratios[i] = min(1, scaled_ratios[i])
        cfg = (base_channels * scaled_ratios).astype(int).tolist()
        for i in range(len(cfg)):
            cfg[i] = _make_divisible(cfg[i], div) # 8 divisible channels
        current_flops = model.cfg2flops(cfg)
        if cnt > 20:
            break
        if abs(current_flops - flops) / flops < 0.01:
            break
        if p:
            print(str(current_flops)+'---'+str(flops)+'---left: '+str(left)+'---right: '+str(right)+'---cfg: '+str(cfg))
        if current_flops < flops:
            left = scale
        elif current_flops > flops:
            right = scale
        else:
            break
    return cfg

def weight2mask(weight, keep_c): # simple L1 pruning
    weight_copy = weight.abs().clone() #权重的绝对值
    L1_norm = torch.sum(weight_copy, dim=(1,2,3)) #对于一个weight，得到每个filter的sum
    arg_max = torch.argsort(L1_norm, descending=True) #降序排列得到索引
    arg_max_rev = arg_max[:keep_c].tolist() #keep_c是要保留多少out_channel
    mask = np.zeros(weight.shape[0]) #生成一个shape为out_channel的mask
    mask[arg_max_rev] = 1 #把mask中要保留的channel部分的值设为1
    return mask

def get_unpruned_weights(model, model_origin): #使用mask来生成新的conv和bn层，model是剪枝过但没有权重的模型，model_origin是原模型，且有权重
    #传入时，两个model都在cpu上
    #要保证两个model中的模块是一一对应的
    masks = [] #存放很多mask的list
    for [m0, m1] in zip(model_origin.named_modules(), model.named_modules()):
        if isinstance(m0[1], nn.Conv2d): #卷积层
            if m0[1].weight.data.shape!=m1[1].weight.data.shape: #对应卷积层的shape不相同
                flag = False
                if m0[1].weight.data.shape[1]!=m1[1].weight.data.shape[1]: #in_channels维度不同
                    assert len(masks)>0, "masks is empty!"
                    if m0[0].endswith('downsample.conv'): #降采样卷积
                        if model.config['depth']>=50:
                            mask = masks[-4]
                        else:
                            mask = masks[-3]
                    else: #普通卷积
                        mask = masks[-1]
                    idx = np.squeeze(np.argwhere(mask))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0[1].weight.data[:, idx.tolist(), :, :].clone()
                    flag = True
                    if m0[1].weight.data.shape[0]==m1[1].weight.data.shape[0]: 
                        masks.append(None)
                if m0[1].weight.data.shape[0]!=m1[1].weight.data.shape[0]: #out_channels维度不同
                    if m0[0].endswith('downsample.conv'):
                        mask = masks[-1]
                    else:
                        if flag:
                            mask = weight2mask(w.clone(), m1[1].weight.data.shape[0])
                        else:
                            mask = weight2mask(m0[1].weight.data, m1[1].weight.data.shape[0]) #原模型的权重，需要保留的filter数
                    idx = np.squeeze(np.argwhere(mask))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    if flag:
                        w = w[idx.tolist(), :, :, :].clone() #得到新权重
                    else:
                        w = m0[1].weight.data[idx.tolist(), :, :, :].clone() #得到新权重
                    m1[1].weight.data = w.clone() #赋值
                    masks.append(mask)
                continue
            else: #对应卷积层的shape相同
                m1[1].weight.data = m0[1].weight.data.clone()
                masks.append(None)
        elif isinstance(m0[1], nn.BatchNorm2d): #bn层
            assert isinstance(m1[1], nn.BatchNorm2d), "There should not be bn layer here."
            if m0[1].weight.data.shape!=m1[1].weight.data.shape: #shape不相同
                mask = masks[-1]
                idx = np.squeeze(np.argwhere(mask))#argwhere返回mask中非0元素的索引，squeeze把shape中为1的维度去掉，这样idx就是一个存放mask中非0元素索引的数组
                if idx.size == 1: #idx中只有1个元素
                    idx = np.resize(idx, (1,))
                #把对应的weight,bias,mean,var复制到新模型中
                m1[1].weight.data = m0[1].weight.data[idx.tolist()].clone()
                m1[1].bias.data = m0[1].bias.data[idx.tolist()].clone()
                m1[1].running_mean = m0[1].running_mean[idx.tolist()].clone()
                m1[1].running_var = m0[1].running_var[idx.tolist()].clone()
                continue
            #shape相同就直接复制
            m1[1].weight.data = m0[1].weight.data.clone()
            m1[1].bias.data = m0[1].bias.data.clone()
            m1[1].running_mean = m0[1].running_mean.clone()
            m1[1].running_var = m0[1].running_var.clone()

# noinspection PyUnresolvedReferences
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1): #train_one_epoch中用到
    logsoftmax = nn.LogSoftmax(dim=1)
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def count_parameters(model): #print_net_info中用到，打印网络参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
