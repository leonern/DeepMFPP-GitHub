import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import logging
import math

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel,int(channel//reduction),bias=False), 
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction),channel,bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7,
                                                 stride=1, padding=3, bias=False),
                                               )


    def forward(self, x):
        bahs, chs, _, _ = x.size()     #16 16 24 42

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_avg = self.avg_pool(x).view(bahs, chs)                       
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)  
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add=chn_avg+chn_max
        chn_add=self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add)

        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)

        spa_add = self.spatial_excitation(cat)
        spa_add = self.sigmoid(spa_add)
        spa_cbam = torch.mul(chn_cbam, spa_add)

        return spa_cbam

class CBAMBlock1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel,int(channel//reduction),bias=False), 
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction),channel,bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()

        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                               )
        self.spatial_excitation1d = nn.Sequential(nn.Conv1d(64,64, kernel_size=3,
                                                 stride=1, padding=1, bias=False),#, bias=False
                                                 nn.LeakyReLU()
                                               )

    def forward(self, x):
        bahs, chs, _, _ = x.size()   # [B,IN_CHANS,F,S]

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_avg = self.avg_pool(x).view(bahs, chs)                       
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)  
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add = chn_avg+chn_max
        chn_add = self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add) # [B,IN_CHANS,F,S]

        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = avg_out + max_out
        cat = cat.squeeze() # [B,F,S]

        spa_add = self.spatial_excitation1d(cat)
        spa_add = self.sigmoid(spa_add)
        spa_add = spa_add.unsqueeze(1)
        spa_cbam = torch.mul(chn_cbam, spa_add)

        return spa_cbam # [B,IN_CHANS,F,S]
    
# ####################################################################################################################################################
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)# B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])# B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        #B, C, H, W
        return x

# ############################################################################################################################################################
# SCCONV
import torch
import torch.nn.functional as F  # 导入 PyTorch 的函数库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, group_num:int = 16, eps:float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num  = group_num  # 设置分组数量
        self.gamma      = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta       = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps        = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W  = x.size()  # 获取输入张量的尺寸
        x           = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean        = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std         = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x           = (x - mean) / (std + self.eps)  # 应用批量归一化
        x           = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.gamma + self.beta  # 返回归一化后的张量

# 自定义 SRU（Spatial and Reconstruct Unit）类
class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int,  # 输出通道数
                 group_num:int = 16,  # 分组数，默认为16
                 gate_treshold:float = 0.5,  # 门控阈值，默认为0.5
                 torch_gn:bool = False  # 是否使用PyTorch内置的GroupNorm，默认为False
                 ):
        super().__init__()  # 调用父类构造函数

         # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold  = gate_treshold  # 设置门控阈值
        self.sigomid        = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        gn_x        = self.gn(x)  # 应用分组批量归一化
        w_gamma     = self.gn.gamma / sum(self.gn.gamma)  # 计算 gamma 权重
        reweights   = self.sigomid(gn_x * w_gamma)  # 计算重要性权重

        # 门控机制
        info_mask    = reweights >= self.gate_treshold  # 计算信息门控掩码
        noninfo_mask = reweights < self.gate_treshold  # 计算非信息门控掩码
        x_1          = info_mask * x  # 使用信息门控掩码
        x_2          = noninfo_mask * x  # 使用非信息门控掩码
        x            = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 拆分特征为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 拆分特征为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接

# 自定义 CRU（Channel Reduction Unit）类
class CRU(nn.Module):
    def __init__(self, op_channel:int, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.up_channel     = up_channel = int(alpha * op_channel)  # 计算上层通道数
        self.low_channel    = low_channel = op_channel - up_channel  # 计算下层通道数
        self.squeeze1       = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.squeeze2       = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层

        # 上层特征转换
        self.GWC            = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)  # 创建卷积层
        self.PWC1           = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)  # 创建卷积层

        # 下层特征转换
        self.PWC2           = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)  # 创建卷积层
        self.advavg         = nn.AdaptiveAvgPool2d(1)  # 创建自适应平均池化层

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

# 自定义 ScConv（Squeeze and Channel Reduction Convolution）模型
class ScConv(nn.Module):
    def __init__(self, op_channel:int, group_num:int = 16, gate_treshold:float = 0.5, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数

        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)  # 创建 CRU 层

    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x