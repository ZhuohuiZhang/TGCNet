# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 14:02
# @Author  : Zhuohui Zhang
# @File    : multi_head_attention.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.utils import transpose_input, transpose_output


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, keys, queries, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens * num_heads, bias=False)
        self.w_k = nn.Linear(key_size, num_hiddens * num_heads, bias=False)
        self.w_v = nn.Linear(value_size, num_hiddens * num_heads, bias=False)
        self.w_o = nn.Linear(num_hiddens * num_heads, query_size, bias=False)

    def forward(self, queries, keys, values):
        queries = transpose_input(self.w_q(queries), self.num_heads)
        values = transpose_input(self.w_v(values), self.num_heads)
        keys = transpose_input(self.w_k(keys), self.num_heads)
        output = self.attention(queries, values, keys)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)
