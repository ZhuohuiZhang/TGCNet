# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 20:03
# @Author  : Zhuohui Zhang
# @File    : graph_convolution.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn
import torch


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, adjacency, input_feature):
        support = self.embedding(input_feature).reshape(-1, adjacency.shape[-1], self.output_dim)
        output = torch.bmm(adjacency, support).reshape(-1, self.output_dim)
        return output
