import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import copy
import datetime
import csv
import codecs

from Experiments_RL.experiment_base import HybridBase
from Utils_RL.models import PASAC_QNetwork_MLP, PASAC_PolicyNetwork_MLP
from Utils_RL.utils import copy_param, soft_update, \
    v2id, ReplayBuffer_MLP, SPER_MLP


class PASAC_Agent_MLP(HybridBase):
    def __init__(self, debug, weights, gamma, replay_buffer_size, max_steps,
                 hidden_size, value_lr, policy_lr, batch_size, state_dim,
                 action_discrete_dim, action_continuous_dim,
                 soft_tau, use_exp, reward_l=0, reward_h=1, data_bin=8):
        super(PASAC_Agent_MLP, self).__init__(debug, weights, gamma, replay_buffer_size, max_steps,
                                             hidden_size, value_lr, policy_lr, batch_size, state_dim,
                                             action_discrete_dim, action_continuous_dim)
        assert debug['replay_buffer'] in ['r', 'p']
        if debug['replay_buffer'] == 'r':
            self.replay_buffer = ReplayBuffer_MLP(replay_buffer_size)
        elif debug['replay_buffer'] == 'p':
            self.replay_buffer = SPER_MLP(replay_buffer_size, 'uniform', reward_l, reward_h, data_bin)

        self.soft_q_net1 = PASAC_QNetwork_MLP(max_steps, 
                                              state_dim, 
                                              action_discrete_dim, 
                                              action_continuous_dim,
                                              hidden_size=hidden_size, 
                                              batch_size=batch_size).to(self.device)
        self.soft_q_net2 = PASAC_QNetwork_MLP(max_steps, 
                                              state_dim, 
                                              action_discrete_dim, 
                                              action_continuous_dim,
                                              hidden_size=hidden_size, 
                                              batch_size=batch_size).to(self.device)
        self.target_soft_q_net1 = PASAC_QNetwork_MLP(max_steps, 
                                                     state_dim, 
                                                     action_discrete_dim, 
                                                     action_continuous_dim,
                                                     hidden_size=hidden_size, 
                                                     batch_size=batch_size).to(self.device)
        self.target_soft_q_net2 = PASAC_QNetwork_MLP(max_steps, 
                                                     state_dim, 
                                                     action_discrete_dim, 
                                                     action_continuous_dim,
                                                     hidden_size=hidden_size, 
                                                     batch_size=batch_size).to(self.device)
        self.policy_net = PASAC_PolicyNetwork_MLP(state_dim,
                                                  max_steps,
                                                  action_discrete_dim,
                                                  action_continuous_dim,
                                                  hidden_size=hidden_size,
                                                  batch_size=batch_size).to(self.device)
                                                  
        self.log_alpha_c = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.log_alpha_d = torch.tensor(
            [-1.6094], dtype=torch.float32, requires_grad=True, device=self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.models = {'policy': self.policy_net,
                       'value1': self.soft_q_net1, 'target_value1': self.target_soft_q_net1,
                       'value2': self.soft_q_net2, 'target_value2': self.target_soft_q_net2}

        self.soft_q_optimizer1 = torch.optim.Adam(
            self.soft_q_net1.parameters(), lr=value_lr, weight_decay=debug['L2_norm'])
        self.soft_q_optimizer2 = torch.optim.Adam(
            self.soft_q_net2.parameters(), lr=value_lr, weight_decay=debug['L2_norm'])
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_lr, weight_decay=debug['L2_norm'])

        self.alpha_optimizer_d = torch.optim.Adam(
            [self.log_alpha_d], lr=debug['alpha_lr'], weight_decay=self.debug['L2_norm'])
        self.alpha_optimizer_c = torch.optim.Adam(
            [self.log_alpha_c], lr=debug['alpha_lr'], weight_decay=self.debug['L2_norm'])

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.max_steps = max_steps
        self.DETERMINISTIC = False
        self.soft_tau = soft_tau
        self.use_exp = use_exp

        # load models if needed
        if debug['load_model'] and debug['load_filename'] is not None:
            self.load(None)

    def act(self, state:np.ndarray, sampling:bool=False):
        """
        explore with original action and return one-hot action encoding
        Note - 'sampling' = sample an action tuple with the probability = <normalized action embedding>
        """
        param, action_v = self.policy_net.get_action(state, deterministic=self.DETERMINISTIC)
        action_enc = (action_v, param)
        if sampling:
            a = np.log(action_v) / 1.0
            dist = np.exp(a) / np.sum(np.exp(a))
            choices = range(len(a))
            action = np.random.choice(choices, p=dist), param
        else:
            action = v2id((action_v, param), from_tensor=False)
        return action, action_enc, action_v, param

    def update(self, batch_size, auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2,
               need_print=False, ep=None):
        if isinstance(self.replay_buffer, ReplayBuffer_MLP):
            state, action_v, param, reward, next_state, done = self.replay_buffer.sample(batch_size)
        elif isinstance(self.replay_buffer, SPER_MLP):
            batch, indices, weights = self.replay_buffer.sample(batch_size)
            state, action_v, param, reward, next_state, done, _ = batch
            weights = torch.FloatTensor(weights).unsqueeze(-1).unsqueeze(-1).to(self.device)
        else:
            raise NotImplementedError

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action_v   = torch.FloatTensor(action_v).to(self.device)
        param      = torch.FloatTensor(param).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # [predict]
        predicted_q_value1 = self.soft_q_net1(state, action_v, param)
        predicted_q_value2 = self.soft_q_net2(state, action_v, param)
        new_action_v, new_param, log_prob, _, _, _ = self.policy_net.evaluate(state)
        new_next_action_v, new_next_param, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # -----Updating alpha wrt entropy-----
        # [compute action log prob]
        def get_log_action_prob(action_prob, use_exp):
            if self.use_exp:  # expectation
                action_log_prob = torch.log(action_prob)
                action_log_prob = action_log_prob.mul(action_prob)  # calculate expectation
                action_log_prob[action_log_prob!=action_log_prob] = 0  # set NaN to zero
                action_log_prob.clamp_(-10, 0)
                action_sum_log_prob = torch.sum(action_log_prob, dim=-1)
                action_sum_log_prob = action_sum_log_prob.view(action_sum_log_prob.size(0), 1)
            else:  # sampling
                action_sample_all = []
                for action in action_prob:
                    action_sample = []
                    for a in action:
                        a = a.detach().cpu().numpy()
                        choices = range(len(a))
                        if a[0] == 0:
                            action_sample.append(0)
                            continue
                        idx = np.random.choice(choices, p=a)
                        action_sample.append(a[idx].item())
                    action_sample_all.append(action_sample)
                action_sample_all = torch.FloatTensor(action_sample_all)
                action_log_prob = torch.log(action_sample_all)
                action_sum_log_prob = action_log_prob.view(action_log_prob.size(0),
                                                           self.max_steps, 1)
                action_sum_log_prob = action_sum_log_prob.to(self.device)

            return action_sum_log_prob
        
        action_sum_log_prob = get_log_action_prob(new_action_v, self.use_exp)
        action_sum_log_prob_next = get_log_action_prob(new_next_action_v, self.use_exp)

        # [update temperature alpha]
        if auto_entropy:
            alpha_loss_d = -(self.log_alpha_d * (action_sum_log_prob +
                                             target_entropy).detach())
            alpha_loss_d = alpha_loss_d.mean()
            self.alpha_optimizer_d.zero_grad()
            alpha_loss_d.backward()
            self.alpha_optimizer_d.step()
            self.alpha_d = self.log_alpha_d.exp()

            alpha_loss_c = -(self.log_alpha_c * (log_prob +
                                             target_entropy).detach())
            alpha_loss_c = alpha_loss_c.mean()
            self.alpha_optimizer_d.zero_grad()
            alpha_loss_c.backward()
            self.alpha_optimizer_c.step()
            self.alpha_c = self.log_alpha_c.exp()
        else:
            self.alpha_c = 1.
            self.alpha_d = 1.
            alpha_loss_d = 0
            alpha_loss_c = 0
        if need_print:
            print('[debug: alpha_c]', self.alpha_c.data[0].item())
            print('[debug: alpha_d]', self.alpha_d.data[0].item())

        # [compute value loss]
        predict_target_q1 = self.target_soft_q_net1(
            next_state, new_next_action_v, new_next_param)
        predict_target_q2 = self.target_soft_q_net2(
            next_state, new_next_action_v, new_next_param)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - \
            self.alpha_c * next_log_prob - self.alpha_d * action_sum_log_prob_next
        target_q_value = reward + (1 - done) * gamma * target_q_min

        q_value_loss1_elementwise = predicted_q_value1 - target_q_value.detach()
        q_value_loss2_elementwise = predicted_q_value2 - target_q_value.detach()

        # [compute policy loss]
        predict_q1 = self.soft_q_net1(state, new_action_v, new_param)
        predict_q2 = self.soft_q_net2(state, new_action_v, new_param)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        policy_loss_elementwise = self.alpha_c * log_prob + \
            self.alpha_d * action_sum_log_prob - predicted_new_q_value

        # [compute total loss]
        if isinstance(self.replay_buffer, SPER_MLP):
            td_pl = (q_value_loss1_elementwise.abs() +
                          q_value_loss2_elementwise.abs() +
                          policy_loss_elementwise.abs()).sum(dim=1)
            q_value_loss1 = (q_value_loss1_elementwise ** 2 * weights).mean()
            q_value_loss2 = (q_value_loss2_elementwise ** 2 * weights).mean()
            policy_loss = (policy_loss_elementwise * weights).mean()
        else:
            q_value_loss1 = self.soft_q_criterion1(q_value_loss1_elementwise, torch.zeros_like(
                q_value_loss1_elementwise))
            q_value_loss2 = self.soft_q_criterion2(q_value_loss2_elementwise, torch.zeros_like(
                q_value_loss2_elementwise))
            policy_loss = policy_loss_elementwise.mean()

        # [update networks]
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.soft_q_net1.parameters(), 5)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.soft_q_net2.parameters(), 5)
        self.soft_q_optimizer2.step()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 5)
        self.policy_optimizer.step()

        # [soft update the target value net]
        soft_update(self.target_soft_q_net1, self.soft_q_net1, soft_tau)
        soft_update(self.target_soft_q_net2, self.soft_q_net2, soft_tau)

        # [update priorities]
        if isinstance(self.replay_buffer, SPER_MLP):
            self.replay_buffer.priority_update(indices, td_pl.reshape(batch_size).tolist())

        return predicted_new_q_value.mean()
