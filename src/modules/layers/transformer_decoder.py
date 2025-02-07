# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 18:59
# @Author  : Zhuohui Zhang
# @File    : transformer_decoder.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn
from modules.layers.multi_head_attention import MultiHeadAttention
from modules.layers.add_norm import AddNorm
from modules.layers.position_wise_ffn import PositionWiseFFN


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_input)
        self.add_norm_2 = AddNorm(norm_shape, dropout)

    def forward(self, x):
        att_out = self.attention(x, x, x)
        norm_out_1 = self.add_norm_1(x, att_out)
        norm_out_2 = self.add_norm_2(norm_out_1, self.ffn(norm_out_1))
        return norm_out_2


class TransformerDecoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module("block" + str(i),
                                   DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                ffn_num_input, ffn_num_hiddens, num_heads, dropout))

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        return x
