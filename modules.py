from multiprocessing import pool
from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


# GAT
#############################################################################################################################################
def adjConcat(a, b):
    """
    Combine the two matrices a,b diagonally along the diagonal direction and fill the empty space with zeros
    """
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  
    right = np.row_stack((np.zeros((lena, lenb)), b))  
    result = np.hstack((left, right)) 
    return result

def final_adj(counts,adj_matrix):
    for i in range(counts):
            if i == 0:
                end_adj_matrix = adj_matrix.cpu().detach().numpy()
            else:
                end_adj_matrix = adjConcat(end_adj_matrix, adj_matrix.cpu().detach().numpy()) 
    return end_adj_matrix

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        self.attentions2 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        attention_matrix_list = list()   # Lists for saving attention: len(attention_matrix_list) = layer_num * head
        # First layer gat
        for idx, att in enumerate(self.attentions1):
            x1, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x1
            else:
                x_tmp = torch.cat((x_tmp, x1), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        # Second layer gat
        for idx, att in enumerate(self.attentions2):
            x2, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x2
            else:
                x_tmp = torch.cat((x_tmp, x2), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        return x, attention_matrix_list

# transformer modules
###########################################################################################################################
class AddNorm(nn.Module):
    """残差连接后进行层归一化"""

    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, x, y):
        return  self.ln(x + self.dropout(y)) 


class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""

    def __init__(self, ffn_input, ffn_hiddens,mlp_bias=True):
        super(PositionWiseFFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input, ffn_hiddens, bias=mlp_bias),
            nn.ReLU(),
            nn.Linear(ffn_hiddens, ffn_input, bias=mlp_bias),
        )

    def forward(self, x):
        return self.ffn(x)
    
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
class AttentionEncode(nn.Module):

    def __init__(self, dropout, embedding_size, num_heads,ffn=False):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.seq_len = 50
        self.is_ffn = ffn
        
        self.att = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                         num_heads=num_heads,
                                         dropout=0.6
                                         )
    
        self.addNorm = AddNorm(normalized=[self.seq_len, self.embedding_size], dropout=self.dropout)

        self.FFN = PositionWiseFFN(ffn_input=self.embedding_size, ffn_hiddens=self.embedding_size*2)

    def forward(self, x):
        bs,_,_ = x.size()
        MHAtt, _ = self.att(x, x, x)
        MHAtt_encode = self.addNorm(x, MHAtt)

        if self.is_ffn:
            ffn_in = MHAtt_encode # bs,seq_len,feat_dims
            ffn_out = self.FFN(ffn_in)
            MHAtt_encode = self.addNorm(ffn_in,ffn_out)

        return MHAtt_encode

class FAN_encode(nn.Module):

    def __init__(self, dropout, shape):
        super(FAN_encode, self).__init__()
        self.dropout = dropout
        self.addNorm = AddNorm(normalized=[1, shape], dropout=self.dropout)
        self.FFN = PositionWiseFFN(ffn_input=shape, ffn_hiddens=(2*shape))
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        #x = self.ln(x)
        ffn_out = self.FFN(x)
        encode_output = self.addNorm(x, ffn_out)

        return encode_output