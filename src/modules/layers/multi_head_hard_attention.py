# -*- coding: utf-8 -*-
# @Time    : 2023/12/29 20:57
# @Author  : Zhuohui Zhang
# @File    : multi_head_hard_attention.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.utils import transpose_input, sequence_mask


def mask_gumbel_softmax(x, mask):
    if mask is None:
        return F.gumbel_softmax(x, dim=-1, tau=0.01)[:, :, :, 0]
    else:
        shape = x.shape
        mask = mask.reshape(-1)
        x = sequence_mask(x.reshape(-1, shape[-1]), mask, value=-1e6)
        return F.gumbel_softmax(x.reshape(shape), dim=-1, tau=0.01)[:, :, :, 0]


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = num_hiddens
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 2, bias=False)

    def forward(self, queries, keys, mask):
        queries, keys = self.w_q(queries), self.w_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        weights = mask_gumbel_softmax(scores, mask)

        return weights


class MultiHeadHardAttention(nn.Module):
    def __init__(self, key_size, query_size, num_heads, embed_dim):
        super(MultiHeadHardAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = AdditiveAttention(embed_dim, embed_dim, embed_dim)
        self.w_q = nn.Linear(query_size, embed_dim * num_heads)
        self.w_k = nn.Linear(key_size, embed_dim * num_heads)

    def forward(self, queries, keys, mask):
        queries = transpose_input(self.w_q(queries), self.num_heads)
        keys = transpose_input(self.w_k(keys), self.num_heads)
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=self.num_heads, dim=0)
        hard_attention_weights = self.attention(queries, keys, mask)

        return hard_attention_weights
