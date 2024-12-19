# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 16:19
# @Author  : Zhuohui Zhang
# @File    : position_wise_ffn.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_out_puts):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_out_puts)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
