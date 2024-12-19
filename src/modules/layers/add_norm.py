# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 15:05
# @Author  : Zhuohui Zhang
# @File    : add_norm.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)
