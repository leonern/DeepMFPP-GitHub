import torch
import torch.nn as nn
import torch.nn.functional as F

class Poly_cross_entropy(nn.Module):
    def __init__(self,weight=None,reduction='mean'):
        super(Poly_cross_entropy,self).__init__()

        if weight is None:
            self.cel = nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.cel = nn.CrossEntropyLoss(reduction=reduction,weight=weight)

        self.softmax = nn.Softmax(1)
        
    def forward(self,logits, labels, epsilon=1.0):
        labels = labels.to(torch.float)
        poly1 = torch.sum(labels * self.softmax(logits), dim=-1)
        ce_loss = self.cel(logits, labels)
        pce_loss = ce_loss + epsilon * (1 - poly1)

        return torch.mean(pce_loss)

class Poly_BCE(nn.Module):
    def __init__(self,weight=None,logits=False,reduction='mean'):
        super(Poly_BCE,self).__init__()

        self.cel = nn.BCELoss(reduction=reduction,weight=weight)
        self.logits = logits
        self.reduction = reduction
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,logits, labels, epsilon=1.0):
        labels = labels.to(torch.float)
        poly1 = torch.sum(labels * self.softmax(logits), dim=-1)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(logits, labels, reduction=self.reduction)
        pce_loss = BCE_loss + epsilon * (1 - poly1)

        return torch.mean(pce_loss)
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        
        return loss_contrastive

class InfoNCELoss(nn.Module):
    def __init__(self, temperature):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature # 温度参数

    def forward(self, q, k):
        # q: query向量，shape为(batch_size, dim)
        # k: key向量，shape为(batch_size, dim)
        # 对q和k向量进行L2正则化，即除以它们的L2范数
        q = q / torch.norm(q, dim=1, keepdim=True)
        k = k / torch.norm(k, dim=1, keepdim=True)
        # 计算query和key的点积，shape为(batch_size, batch_size)
        logits = torch.matmul(q, k.t())

        # 对角线元素为正样本的得分，其他元素为负样本的得分
        # 将对角线元素提取出来，shape为(batch_size,)
        pos = torch.diag(logits)

        # 用指数函数对所有得分进行缩放，除以温度参数
        logits = torch.exp(logits / self.temperature)
        pos = torch.exp(pos / self.temperature)

        # 计算分母，即所有样本的得分之和，减去对角线元素，shape为(batch_size,)
        den = logits.sum(dim=1) - pos

        # 计算分子，即正样本的得分，shape为(batch_size,)
        num = pos

        # 计算损失，即负对数似然，取平均值，shape为标量
        loss = -torch.log(num / den).mean()

        return loss
    
# *******************************************************************************************************************************************************************
import torch
import torch.nn as nn
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size:int=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        N = 2 * self.batch_size * self.world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size * self.world_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N . 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0) 
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# *********************************************************************************************************************************************************************
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.CELoss = torch.nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
       
        return F_loss
    
class AsymmetricLoss(nn.Module):
    """Asymmetric loss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 reduction='mean'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss"""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        loss = 1 - (2 * num) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
    
class FocalDiceLoss(nn.Module):
    """Multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
