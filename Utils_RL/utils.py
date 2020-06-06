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

class PrioritizedReplayBuffer_Original:
    """ 
    Prioritized Replay Buffer see 
    https://arxiv.org/pdf/1511.05952.pdf and https://cardwing.github.io/files/RL_course_report.pdf for details
    """
    e = 0.01
    d = 100
    alpha = 2.0
    beta = 1.0
    beta_increment_per_sampling = 0.01

    def __init__(self, capacity):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self.buffer = SumTree(capacity)
        self._max_priority = 1.0

    def _get_priority(self, error):
        """ get priority for TD error"""
        return (np.abs(error) + self.e + self.d) ** self.alpha

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        self.buffer.add(self._max_priority, Experience(state, action, last_action, reward, next_state, done, episode, 0, 0))

    def priority_update(self, indices, priorities, tds):
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
        for i in range(len(indices)):
            self.buffer.data[indices[i]-self.buffer.capacity+1] = self.buffer[indices[i]-self.buffer.capacity+1]._replace(td=tds[i], st=self.buffer[indices[i]-self.buffer.capacity+1].st+1)

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """
        batch = []
        indices = []
        priorities = []
        weights = []
        segment = self.buffer.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            # do not get zero from SumTree
            while True:
                a = segment * i
                b = segment * (i + 1)
                r = random.uniform(a, b)
                index, priority, data = self.buffer.get(r)
                if data:
                    break
            priorities.append(priority)
            weights.append((1./self.capacity/priority) **
                           self.beta if priority > 1e-16 else 0)
            indices.append(index)
            batch.append(data)  # [batch_size, max_step, 6(state, action_v, param, reward, next_state, done), *]

        weights = np.array(weights)

        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []
        for sample in batch:
            state, action, last_action, reward, next_state, done, episode, _, _= sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            episode_lst.append(episode)

        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices, weights

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer.data[key]

    @property
    def priorities(self):
        return self.buffer.tree[self.capacity-1:self.capacity-1+len(self)]

    def print_status(self):
        for i in range(len(self)):
            experience = self[i]
            print(i, 'rw: ', str(experience.rew), 'td: ', str(experience.td), 'ep: ', str(experience.episode), 'st: ', str(experience.st))


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


class SPER_MLP:
    """ 
    Implementation of SPER in our paper
    """

    def __init__(self, capacity, capacity_distribution, reward_l, reward_h, data_bin):
        """ Prioritized experience replay buffer initialization."""
        self.capacity = capacity
        self._max_priority = 1.0
        self.bins = []
        self.data_bin = data_bin
        self.capacity_distribution=capacity_distribution
        self.reward_l = reward_l
        self.reward_h = reward_h
        self.interval = (self.reward_h-self.reward_l)/self.data_bin
        if capacity_distribution=='uniform':
            for _ in range(self.data_bin):
                self.bins.append(PrioritizedReplayBuffer_Original(capacity//self.data_bin))
        elif capacity_distribution=='exponential':
            for i in range(self.data_bin):
                self.bins.append(PrioritizedReplayBuffer_Original(capacity//(2**(self.data_bin-i))))

    def push(self, state, action, last_action, reward, next_state, done, episode):
        """  push a sample into prioritized replay buffer"""
        bin_id = int(min(max((reward-self.reward_l)//self.interval,0),self.data_bin-1))
        self.bins[bin_id].push(state, action, last_action, reward, next_state, done, episode)
    
    @property
    def buffer(self):
        return list(itertools.chain.from_iterable([_bin[:len(_bin)] for _bin in self.bins]))

    @property
    def bin_size(self):
        return [len(b) for b in self.bins]
    
    def priority_update(self, indices, priorities, tds):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for idx, error, td in zip(indices, priorities, tds):
            if idx == -1:
                continue
            if self.capacity_distribution=='uniform':
                bin_id = idx//(self.capacity//self.data_bin*2-1)
            elif self.capacity_distribution == 'exponential':
                bin_id = 0
                capacity_sum = 0
                while capacity_sum+self.bins[bin_id].buffer.tree_size<=idx:
                    capacity_sum+=self.bins[bin_id].buffer.tree_size
                    bin_id+=1
            idx_in_bin = idx - sum([abin.buffer.tree_size for abin in self.bins[:bin_id]])
            if idx_in_bin<0:
                print(idx_in_bin)
            self.bins[bin_id].priority_update([idx_in_bin],[error],[td])

    def sample(self, batch_size):
        """ sample batch_size data from replay buffer """

        assert isinstance(batch_size, int) or isinstance(batch_size, np.int64) or len(batch_size)==self.data_bin, "Batch size must be a number or a list as long as the list of bins"
        indices = []
        weights = []
        s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst = [], [], [], [], [], [], []

        if isinstance(batch_size, int) or isinstance(batch_size, np.int64):
            i = -1
            while batch_size>0:
                i=(i+1)%self.data_bin
                if len(self.bins[i])==0: 
                    continue
                _batch, _indices, _weights = self.bins[i].sample(1)
                _indices = (np.array(_indices) + sum([abin.buffer.tree_size for abin in self.bins[:i]])).tolist()
                _s_lst, _a_lst, _la_lst, _r_lst, _ns_lst, _d_lst, _episode_lst = _batch
                s_lst = s_lst + _s_lst
                a_lst = a_lst + _a_lst
                la_lst = la_lst + _la_lst
                r_lst = r_lst + _r_lst
                ns_lst = ns_lst + _ns_lst
                d_lst = d_lst + _d_lst
                indices = indices + _indices
                weights.append(_weights)
                episode_lst = episode_lst + _episode_lst
                batch_size -= 1
        else:
            for i in range(self.data_bin):
                _batch, _indices, _weights = self.bins[i].sample(batch_size[i])
                _indices = (np.array(_indices) + sum([abin.buffer.tree_size for abin in self.bins[:i]])).tolist()
                _s_lst, _a_lst, _la_lst, _r_lst, _ns_lst, _d_lst, _episode_lst = _batch
                s_lst = s_lst + _s_lst
                a_lst = a_lst + _a_lst
                la_lst = la_lst + _la_lst
                r_lst = r_lst + _r_lst
                ns_lst = ns_lst + _ns_lst
                d_lst = d_lst + _d_lst
                indices = indices + _indices
                weights.append(_weights)
                episode_lst = episode_lst + _episode_lst
        weights = np.concatenate(weights)
        return (s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst, episode_lst), indices, weights

    def __len__(self):
        return sum([len(_bin.buffer) for _bin in self.bins])

    def print_status(self):
        for i in range(self.data_bin):
            print('Bin '+str(i))
            self.bins[i].print_status()

    @property
    def priorities(self):
        return list(itertools.chain.from_iterable([_buf.buffer.tree[_buf.capacity-1:_buf.capacity-1+len(_buf)] for _buf in self.bins]))


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
