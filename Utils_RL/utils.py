import copy
import random
from collections import namedtuple

import gym
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Utils_RL.sum_tree import SumTree
import pdb

device = torch.device('cuda')
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATransition = namedtuple('PATransition', ('state', 'action', 'param', 'reward', 'next_state', 'done'))  # ,


########################## (Parameterized) Action Utilities ##########################


def v2id(action_enc: tuple, from_tensor: bool = False):
    return action_enc[0].argmax().item(), action_enc[1] if not from_tensor else \
        action_enc[1].squeeze(dim=0).detach().cpu().numpy()


def id2v(action: tuple, dim_action: int, return_tensor: bool = False):
    """ convert one action tuple from discrete action id to one-hot encoding """
    one_hot = np.zeros(dim_action)  # number of actions including special actions
    one_hot[action[0]] = 1
    return (torch.Tensor(one_hot) if return_tensor else one_hot,
            torch.Tensor(action[1]) if return_tensor else action[1])


def soft_update(target_net, source_net, soft_tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)


def copy_param(target_net, source_net):  # or copy.deepcopy(source_net)
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)


def padding_and_convert(seq, max_length, is_action=False):
    """padding a batch of sequence to max_length, then convert it into torch.FloatTensor"""
    seq_dim_len = torch.LongTensor([len(s) for s in seq])
    if torch.is_tensor(seq):
        for i, s in enumerate(seq):
            if s.shape[0] < max_length:
                zero_arrays = torch.zeros(max_length - s.shape[0], s.shape[0]).cuda()
                s = torch.cat((s, zero_arrays), 0)
            seq[i] = s
    else:
        for i, s in enumerate(seq):
            s = np.array(s)
            if len(s) < max_length:
                zero_arrays = np.zeros((max_length - len(s), len(s[0])))
                s = np.concatenate((s, zero_arrays), 0)
            seq[i] = s
        seq = torch.FloatTensor(seq).to(d)
    return seq, seq_dim_len


def gen_mask(seqs, seq_dim_len, max_len):
    m = []
    for i, seq in enumerate(seqs):
        m1 = []
        for j in range(max_len):
            if j < seq_dim_len[i]:
                m1.append([1])
            else:
                m1.append([0])
        m.append(m1)
    return m


########################## Experience Replay Buffer ##########################


class ReplayBuffer_LSTM:
    """ 
    Replay buffer for agent with LSTM network additionally using previous action, can be used 
    if the hidden states are not stored (arbitrary initialization of lstm for training).
    And each sample contains the whole episode instead of a single step.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


class PrioritizedReplayBuffer_LSTM:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, np.sum(sequence[3])) for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into bins
        # sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            # for every bin
            for i, reward in reward_sum[data_len // self.data_bin * j:data_len // self.data_bin * (j + 1)]:
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)  # clip
                indices.append(index)
                batch.append(data)
                count += 1
                if count >= batch_size // (2 * self.data_bin):
                    break
                # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  
            # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]

        # Normalize for stability
        weights = np.array(weights)
        weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]


class ReplayBuffer_MLP:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action_v, param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_v, param, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action_v, param, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action_v, param, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer_MLP:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01
    data_bin = 8

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, (state, action, last_action, reward, next_state, done))

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error in zip(indices, priorities):
            p = self._get_priority(error)
            self._max_priority = max(self._max_priority, p)
            self.buffer.update(idx, p)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # make Reward Bin
        data_len = len(self.buffer)
        reward_sum = [(i, sequence[3]) for i, sequence in enumerate(self.buffer[:data_len])]
        reward_sum = sorted(reward_sum, key=lambda x: x[1], reverse=True)
        # sort first, then sequentially split the sorted dataset into bins
        # sample Top batch_size//2 from reward
        for j in range(self.data_bin):
            count = 0
            for i, reward in reward_sum[data_len // self.data_bin * j:data_len // self.data_bin * (j + 1)]:
                # get index, priority, data
                index = i - 1 + self.buffer.capacity
                priority = self.buffer.tree[index]
                data = self.buffer.data[i]
                # append to list
                priorities.append(priority)
                weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)  # clip
                indices.append(index)
                batch.append(data)
                count += 1
                if count >= batch_size // (2 * self.data_bin):
                    break
                # for every bin, fetch batch_size//(2 * self.data_bin) data items

        # random sample
        len_random = batch_size - len(indices)
        for i in range(len_random):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1. / self.capacity / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  
            # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]

        # Normalize for stability
        weights = np.array(weights)
        weights /= max(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst = [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]


########################## Action Wrapper ##########################


class NormalizedHybridActions(gym.ActionWrapper):
    """ Action Normalization: just normalize 2nd element in action tuple (id, (params)) """
    def action(self, action: tuple) -> tuple:  # old gym's Tuple needs extra `.spaces`
        low = self.action_space.spaces[1].low
        high = self.action_space.spaces[1].high

        param = low + (action[1] + 1.0) * 0.5 * (high - low)
        param = np.clip(param, low, high)

        return action[0], param

    def reverse_action(self, action: tuple) -> tuple:
        low = self.action_space.spaces[1].low
        high = self.action_space.spaces[1].high

        param = 2 * (action[1] - low) / (high - low) - 1
        param = np.clip(param, low, high)

        return action[0], param
