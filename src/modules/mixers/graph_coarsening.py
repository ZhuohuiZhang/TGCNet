# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 22:09
# @Author  : Zhuohui Zhang
# @File    : graph_coarsening.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from modules.layers.utils import normalization
from modules.layers.sag_pool import SAGPool


class GraphCoarsening(nn.Module):
    def __init__(self, args):
        super(GraphCoarsening, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_dim = int(np.prod(args.obs_shape))
        self.coarsen = SAGPool(self.input_dim, args.coarsening_embed_dim)
        self.embed_dim = args.mixing_embed_dim
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.input_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.input_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        self.hyper_b_1 = nn.Linear(self.input_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, obs, adjacency):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        normalize_adjacency = normalization(adjacency).reshape(-1, self.n_agents, self.n_agents)
        node_features = obs.reshape(-1, self.input_dim)
        graph_indicator = torch.arange(agent_qs.shape[0], device=agent_qs.device).repeat_interleave(self.n_agents)
        pool_state = self.coarsen(normalize_adjacency, node_features, graph_indicator)

        w1 = torch.abs(self.hyper_w_1(pool_state))
        b1 = self.hyper_b_1(pool_state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(pool_state))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State-dependent bias
        v = self.V(pool_state).view(-1, 1, 1)

        # Compute final output
        y = torch.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
