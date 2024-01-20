import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import logging
import math
from attention import CBAMBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = 0.00000001
def squash(x):
    # not concern batch_size, maybe rewrite
    s_squared_norm = torch.sum(x*x,1,keepdim=True) + epsilon
    scale = torch.sqrt(s_squared_norm)/(1. + s_squared_norm)
    # out = (batch_size,1,10)*(batch_size,16,10) = (batch_size,16,10)
    out = scale * x
    return out

class Res_Net(nn.Module):
    def __init__(self,input_cha,dropout: float=0.6):
        super(Res_Net,self).__init__()
        self.conv1 = nn.Conv2d(input_cha,input_cha,3,padding=1)
        self.conv2 = nn.Conv2d(input_cha,input_cha,5,padding=2)
        self.conv3 = nn.Conv2d(input_cha,input_cha,7,padding=3)

        self.cbamBlock = CBAMBlock(input_cha) 
        self.drop2d = nn.Dropout2d(dropout)
        self.drop = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm2d(input_cha)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()

    def forward(self,x):
        init_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu2(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out += init_x
        out = self.relu2(out)
        out = self.drop2d(out)

        return out
    
     
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(message)s', #-%(name)s
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)

    return logger

def Category_weight1(dataset):
    '''
    dataset = Data.TensorDataset(data,label),是Subset对象,而不是TensorDataset对象 \n
    可以是train_dataset,也可以是test_dataset \n
    利用sum(label_counts) / (num_classes * count)来计算类别权重
    '''
    original_dataset = dataset.dataset
    indices = dataset.indices
    label = original_dataset.tensors[1][indices]
    #计算每个标签的总数：tensor([ 103, 1723,  835,   99,  418,   45, 1083,   75,  760, 1643,  147,  208,196,  593,   87,  257,  344,  253,  176,   76,  531])
    label_counts = torch.sum(label,dim=0) 
    _,num_classes = label.shape
    weights = [sum(label_counts) / (num_classes * count) for count in label_counts]
    weights = torch.tensor(weights, dtype=torch.float)

    return weights

def CW1(dataset):
    '''
    dataset = Data.TensorDataset(data,label),是Subset对象,而不是TensorDataset对象 \n
    可以是train_dataset,也可以是test_dataset \n
    利用sum(label_counts) / (num_classes * count)来计算类别权重
    '''

    label = dataset
    #计算每个标签的总数：tensor([ 103, 1723,  835,   99,  418,   45, 1083,   75,  760, 1643,  147,  208,196,  593,   87,  257,  344,  253,  176,   76,  531])
    label_counts = torch.sum(label,dim=0) 
    _,num_classes = label.shape
    weights = [sum(label_counts) / (num_classes * count) for count in label_counts]
    weights = torch.tensor(weights, dtype=torch.float)

    return weights

def Category_weight2(dataset):
    '''
    dataset = Data.TensorDataset(data,label),是Subset对象,而不是TensorDataset对象 \n
    PrMLTP的权重平衡方法
    '''
    original_dataset = dataset.dataset
    indices = dataset.indices
    label = original_dataset.tensors[1][indices]
    label_counts = torch.sum(label,dim=0)
    total,_ = label.shape
    weights = [5 * math.pow(int((math.log((total / count), 2))), 2) for count in label_counts]
    weights = torch.tensor(weights, dtype=torch.float)

    return weights

def CW2(dataset):
    '''
    dataset = Data.TensorDataset(data,label),是Subset对象,而不是TensorDataset对象 \n
    PrMLTP的权重平衡方法
    '''
    label = dataset
    label_counts = torch.sum(label,dim=0)
    total,_ = label.shape
    weights = [5 * math.pow(int((math.log((total / count), 2))), 2) for count in label_counts]
    weights = torch.tensor(weights, dtype=torch.float)

    return weights

def Category_weight3(dataset):
    '''
    dataset = Data.TensorDataset(data,label),是Subset对象,而不是TensorDataset对象
    '''
    original_dataset = dataset.dataset
    indices = dataset.indices
    label = original_dataset.tensors[1][indices]
    label_counts = torch.sum(label,dim=0)
    total,num_classes = label.shape

    weights1 = [sum(label_counts) / (num_classes * count) for count in label_counts]
    weights2 = [int((math.log((total / count), 2))) for count in label_counts]
    weights = (np.array(weights1)+np.array(weights2))/2
    weights = torch.tensor(weights, dtype=torch.float)

    return weights

class CosineScheduler:
    def __init__(self, max_update: int=10000, base_lr: float=1e-3, 
                 final_lr: float=0, warmup_steps: int=500, warmup_begin_lr: float=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, steps):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(steps-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, steps):
        if steps < self.warmup_steps:
            return self.get_warmup_lr(steps)
        if steps <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (steps-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

