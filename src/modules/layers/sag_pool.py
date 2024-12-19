# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 19:47
# @Author  : Zhuohui Zhang
# @File    : sag_pool.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_convolution import GraphConvolution
from .self_attention_pooling import SelfAttentionPooling
from .utils import global_avg_pool, global_max_pool


class SAGPool(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAGPool, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, input_dim)

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))
        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        pool = self.pool(adjacency, gcn_feature)
        pool_out = torch.cat((global_avg_pool(pool, graph_indicator),
                             global_max_pool(pool, graph_indicator)), dim=1)
        pool_states = F.relu(self.fc1(pool_out))

        return pool_states
