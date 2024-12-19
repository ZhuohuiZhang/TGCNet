# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 20:35
# @Author  : Zhuohui Zhang
# @File    : self_attention_pooling.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn
import torch.nn.functional as F
from .graph_convolution import GraphConvolution


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature):
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = F.tanh(attn_score)
        hidden = input_feature * attn_score.view(-1, 1)
        return hidden
