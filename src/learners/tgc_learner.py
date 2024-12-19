# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 11:01
# @Author  : Zhuohui Zhang
# @File    : tgc_learner.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import copy
import torch
from torch.optim import Adam
from modules.mixers import REGISTRY as mix_REGISTRY
from components.standarize_stream import RunningMeanStd
from components.episode_buffer import EpisodeBatch


class TGCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = mix_REGISTRY[self.args.mixer](args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        self.optimiser = Adam(params=self.params, lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.args.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        weight_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, weights = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            weight_out.append(weights)
        mac_out = torch.stack(mac_out, dim=1)
        weight_out = torch.stack(weight_out, dim=1)[:, :-1]

        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        with torch.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_weight_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, target_weights = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                target_weight_out.append(target_weights)
            target_mac_out = torch.stack(target_mac_out[1:], dim=1)
            target_weight_out = torch.stack(target_weight_out[1:], dim=1)
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["obs"][:, :-1], weight_out)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["obs"][:, 1:], target_weight_out)

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (self.args.target_update_interval_or_tau > 1 and
                (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.pt".format(path))
        torch.save(self.optimiser.state_dict(), "{}/opt.pt".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load("{}/mixer.pt".format(path),
                                                  map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(torch.load("{}/opt.pt".format(path),
                                                  map_location=lambda storage, loc: storage))
