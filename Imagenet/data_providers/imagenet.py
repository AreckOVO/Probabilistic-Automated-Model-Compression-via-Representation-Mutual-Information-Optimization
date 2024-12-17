import os
import shutil
import numpy as np
import torch.utils.data

import utils
from data_providers import DataProvider
import pdb

def make_imagenet_subset(path2subset, n_sub_classes, path2imagenet='/home/data/imagenet'): #创建一个含n_sub_classes个子类的imagenet子集，子集位置存放在path2subset
    imagenet_train_folder = os.path.join(path2imagenet, 'train')#训练集
    imagenet_val_folder = os.path.join(path2imagenet, 'val')#测试集

    subfolders = sorted([f.path for f in os.scandir(imagenet_train_folder) if f.is_dir()])
    #对训练集文件夹进行迭代，获取里面的子目录，结果是一个list，里面都是字符串，格式为'/home/data/imagenet/train/n01440764'
    # np.random.seed(DataProvider.VALID_SEED)
    np.random.shuffle(subfolders)#对每个类的图片随机排序

    chosen_train_folders = subfolders[:n_sub_classes] #子集需要的类个数
    class_name_list = []
    for train_folder in chosen_train_folders:
        class_name = train_folder.split('/')[-1] #得到类名，如：n01440764
        class_name_list.append(class_name) #加入到类list中

    print('=> Start building subset%d' % n_sub_classes)
    for cls_name in class_name_list:
        #拷贝训练集
        src_train_folder = os.path.join(imagenet_train_folder, cls_name) #该类的源目录
        target_train_folder = os.path.join(path2subset, 'train/%s' % cls_name) #该类的目标目录
        shutil.copytree(src_train_folder, target_train_folder) #拷贝文件
        print('Train: %s -> %s' % (src_train_folder, target_train_folder)) #打印
        #拷贝验证集
        src_val_folder = os.path.join(imagenet_val_folder, cls_name) #该类的源目录
        target_val_folder = os.path.join(path2subset, 'val/%s' % cls_name) #该类的目标目录
        shutil.copytree(src_val_folder, target_val_folder) #拷贝文件
        print('Val: %s -> %s' % (src_val_folder, target_val_folder)) #打印
    print('=> Finish building subset%d' % n_sub_classes) #完成子集创建


class ImagenetDataProvider(DataProvider):   #DALI
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=24, manual_seed = 12, load_type='dali', local_rank=0, world_size=1, **kwargs):
        
        self._save_path = save_path
        self.valid = None
        if valid_size is not None:
            pass
        else:
            self.train = utils.get_imagenet_iter(data_type='train', image_dir=self.train_path, #DALI
                                                batch_size=train_batch_size, num_threads=n_worker,
                                                device_id=local_rank, manual_seed=manual_seed,
                                                num_gpus=torch.cuda.device_count(), crop=self.image_size,
                                                val_size=self.image_size, world_size=world_size, local_rank=local_rank)
            self.test = utils.get_imagenet_iter(data_type='val', image_dir=self.valid_path, manual_seed=manual_seed, #DALI
                                                batch_size=test_batch_size, num_threads=n_worker, device_id=local_rank,
                                                num_gpus=torch.cuda.device_count(), crop=self.image_size,
                                                val_size=256, world_size=world_size, local_rank=local_rank)
        if self.valid is None:
            self.valid = self.test
        
    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/data/imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224
