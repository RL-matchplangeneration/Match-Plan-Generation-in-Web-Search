import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
import numpy as np
import pickle

from gym.wrappers.time_limit import TimeLimit

import datetime
import click
import os
import math
import copy
import logging
import codecs

from tensorboardX import SummaryWriter

from Utils_RL.utils import *


class Experiment(object):
    """base class for experiments, initialize, load query and visualize"""

    def __init__(self, debug, weights, gamma, replay_buffer_size, max_steps, hidden_size, batch_size):
        # [settings]
        self.debug = debug
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.total_step = 0.
        self.plot_frequency = 20

        # [environment]
        self.max_steps = max_steps
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.replay_buffer_length = 0

        self.models = {}  # [models] to be defined in the descendant classes

        logging.basicConfig(filename='./' + debug['log'] + '.log',
                            filemode='w+',
                            level=logging.WARNING,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d %I:%M:%S %p")
        self.logger = logging.getLogger("Benchmark")
        self.logger.info("Log Begin")

    def __str__(self):
        super().__str__()

    def save(self, episodes=None):
        timestamp = str(datetime.datetime.now().strftime('%b_%d_%Y_%H_%M_%S'))  # add timestamp to prevent overriding
        save_name = self.debug['log'] + "_" + str(episodes) + "_" + timestamp
        save_path = self.debug['checkpoint_save_path']
        if not os.path.exists(save_path):
            os.system('mkdir ' + save_path)

        save_name_list = ''
        for name, model in self.models.items():
            complete_save_name = save_path + '/' + save_name + "_" + name + ".pt"
            save_name_list += '\n\n' + complete_save_name
            torch.save(obj=model.state_dict(), f=open(complete_save_name, 'wb'))
        print('Save file names: {}'.format(save_name_list))

    def load(self, episodes=None):
        save_path = self.debug['checkpoint_save_path']
        load_name = self.debug['load_filename']
        for name, model in self.models.items():
            complete_load_name = save_path + '/' + load_name + "_" + name + ".pt"
            model.load_state_dict(state_dict=torch.load(complete_load_name))
            print("[loaded " + complete_load_name + ']')


class HybridBase(Experiment):
    def __init__(self, debug, weights, gamma, replay_buffer_size, max_steps,
                 hidden_size, value_lr, policy_lr, batch_size, state_dim,
                 action_discrete_dim, action_continuous_dim):
        super(HybridBase, self).__init__(debug, weights, gamma, replay_buffer_size,
                                         max_steps, hidden_size, batch_size)

        self.action_discrete_dim = action_discrete_dim
        self.state_dim = state_dim
        self.action_continuous_dim = action_continuous_dim