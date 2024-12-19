# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 18:34
# @Author  : Zhuohui Zhang
# @File    : utils.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import torch_scatter


def transpose_input(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x, num_heads):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


def sequence_mask(x, mask, value):
    mask = mask.to(torch.bool)
    x[:, 0][~mask] = value
    return x


def normalization(adjacency):
    tensor_adjacency = []
    for i in range(adjacency.size(0)):
        A_i = adjacency[i]
        tensor_adjacency_i = []
        for j in range(A_i.size(0)):
            A_ij = A_i[j].clone()
            A_ij += torch.eye(A_ij.shape[0], device=A_ij.device)
            degree = A_ij.sum(-1, keepdim=True)
            d_hat = torch.diag(torch.pow(degree, -0.5).flatten())
            tensor_adjacency_ij = torch.matmul(torch.matmul(d_hat, A_ij), d_hat)
            tensor_adjacency_i.append(tensor_adjacency_ij)
        tensor_adjacency.append(torch.stack(tensor_adjacency_i))
    tensor_adjacency = torch.stack(tensor_adjacency)

    return tensor_adjacency


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def corrupt(x, amount):
    noise = torch.randn_like(x)
    return x * (1 - amount) + noise * amount