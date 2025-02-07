# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 21:53
# @Author  : Zhuohui Zhang
# @File    : tgcnet.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.multi_head_hard_attention import MultiHeadHardAttention
from modules.layers.transformer_decoder import TransformerDecoder


class TGCNet(nn.Module):
    def __init__(self, input_shape, args):
        super(TGCNet, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim * args.n_agents, args.hidden_dim)
        self.fc3 = nn.Sequential(nn.Linear(args.hidden_dim * 2, args.hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(args.hidden_dim, args.n_actions))
        self.att_input_shape = args.hidden_dim * args.n_agents
        self.hard_attention = MultiHeadHardAttention(self.att_input_shape, self.att_input_shape,
                                                     args.enc_att_heads, args.att_enc_dim)
        self.decoder = TransformerDecoder(args.hidden_dim, args.hidden_dim, args.hidden_dim,
                                          args.att_dec_dim, args.hidden_dim, args.hidden_dim,
                                          args.hidden_dim * 2, args.dec_att_heads, args.num_layers, args.dropout)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h_out = self.rnn(x, h_in)

        # communicate
        h = h_out.clone()
        h = h.reshape(-1, self.args.n_agents, self.args.hidden_dim)
        comm = h.reshape(-1, 1, self.args.n_agents * self.args.hidden_dim).repeat(1, self.args.n_agents, 1)
        mask = (1 - torch.eye(self.args.n_agents, device=comm.device)).repeat(h.shape[0], 1, 1)

        # hard-attention
        hard_weights = self.hard_attention(comm, comm, mask)
        hard_weights = hard_weights.reshape(-1, self.args.enc_att_heads, self.args.n_agents, self.args.n_agents)
        hard_weights = torch.round(hard_weights).max(dim=1)[0]
        tot_weights = hard_weights + torch.eye(self.args.n_agents, device=comm.device).repeat(h.shape[0], 1, 1)
        repeat_weights = torch.repeat_interleave(tot_weights, repeats=self.args.hidden_dim, dim=-1)
        mask_comm = comm * repeat_weights
        enc_out = F.relu(self.fc2(mask_comm))

        # decoder
        dec_out = self.decoder(enc_out).reshape(-1, self.args.hidden_dim)

        # q
        q = torch.cat((h_out, dec_out), dim=-1)
        q = self.fc3(q)

        return q, h_out, hard_weights
