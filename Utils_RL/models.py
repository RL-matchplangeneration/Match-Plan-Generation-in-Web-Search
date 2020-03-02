import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from Utils_RL.utils import padding_and_convert
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemperatureSoftmax(nn.Softmax):
    def __init__(self, temperature, dim=None):
        self.temperature = temperature
        super(TemperatureSoftmax, self).__init__(dim)

    def forward(self, x):
        return F.softmax(x / self.temperature, self.dim, _stacklevel=5)


########################## PASAC ##########################


class PASAC_PolicyNetwork_LSTM(nn.Module):
    def __init__(self, num_inputs, max_steps, num_d_actions, num_c_actions, hidden_size, action_range=1.,
                 batch_size=512, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """
        :param num_inputs: state dim
        :param num_actions: discrete action dim
        :param num_params: continuous action dim
        """
        super(PASAC_PolicyNetwork_LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.max_steps = max_steps
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(True)
        )

        # recurrent layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bias=True)

        # output layer
        self.action_fc = nn.Sequential(
            nn.Linear(hidden_size, num_d_actions, bias=True)
        )

        # normalize over action dim / add temperature
        self.action_fc_norm = TemperatureSoftmax(temperature=1., dim=2)
        self.action_fc[-1].weight.data.uniform_(-init_w, init_w)

        # init output layer for PASAC
        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_size, num_c_actions, bias=True))
        self.mean_linear[-1].weight.data.uniform_(-init_w, init_w)
        self.mean_linear[-1].bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Sequential(
            nn.Linear(hidden_size, num_c_actions, bias=True))
        self.log_std_linear[-1].weight.data.uniform_(-init_w, init_w)
        self.log_std_linear[-1].bias.data.uniform_(-init_w, init_w)

    def init_hidden_states(self, bsize):
        hx = torch.zeros(1, bsize, self.hidden_size).float().to(d)
        cx = torch.zeros(1, bsize, self.hidden_size).float().to(d)
        return hx, cx

    def forward(self, state, hidden_in, padding=True, seq_dim_len=None):
        if padding:
            _batch_size = len(state)
            state_padded, _ = padding_and_convert(state, self.max_steps)
            hidden = self.net(state_padded)
            hidden = hidden.view(_batch_size, -1, self.hidden_size)
            hidden_packed = pack_padded_sequence(
                hidden, seq_dim_len, enforce_sorted=False, batch_first=True)

            lstm_out = self.lstm(hidden_packed, hidden_in)

            out_packed = lstm_out[0]
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=self.max_steps)
            hx = lstm_out[1][0]
            cx = lstm_out[1][1]
        else:
            _batch_size = state.shape[0]
            hidden = self.net(state)
            hidden = hidden.view(_batch_size, -1, self.hidden_size)

            lstm_out = self.lstm(hidden, hidden_in)
            out = lstm_out[0]
            hx = lstm_out[1][0]
            cx = lstm_out[1][1]

        d_action_fc = self.action_fc(out)
        d_action_prob = self.action_fc_norm(d_action_fc)

        mean = (self.mean_linear(out))
        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # temperature softmax
        return mean, log_std, d_action_prob, (hx, cx)

    def evaluate(self, state, hidden_in, epsilon=1e-6, seq_dim_len=None):
        '''
        generate sampled action with state as input wrt the policy network
        '''
        mean, log_std, d_action_prob, (hx, cx) = self.forward(
            state, hidden_in, seq_dim_len=seq_dim_len)

        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(d)
        
        c_action_0 = torch.tanh(mean + std * z)
        c_action = self.action_range * c_action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(
            1. - c_action_0.pow(2) + epsilon) - np.log(self.action_range)

        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return d_action_prob, c_action, log_prob, z, mean, log_std, (hx, cx)

    def get_action(self, state, hidden_in, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(d)
        mean, log_std, d_action_prob, (hx, cx) = self.forward(state, hidden_in, padding=False)

        mean = mean[0]  # unsqueeze
        log_std = log_std[0]

        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(d)

        c_action = self.action_range * torch.tanh(mean + std * z)
        c_action = (self.action_range * torch.tanh(mean).detach().cpu().numpy()
                    if deterministic else c_action.detach().cpu().numpy())

        return c_action[0], d_action_prob[0][0].detach().cpu().numpy(), (hx, cx)


class PASAC_QNetwork_LSTM(nn.Module):
    def __init__(self, max_steps, num_inputs, num_d_actions, num_c_actions, hidden_size, batch_size=512, init_w=3e-3):
        """
        :param num_inputs: state dim
        :param num_actions: discrete action dim
        :param num_params: continuous action dim
        """
        super(PASAC_QNetwork_LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.num_d_actions = num_d_actions
        self.num_c_actions = num_c_actions

        self.max_steps = max_steps

        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_d_actions + \
                      num_c_actions, hidden_size, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(True)
        )

        # recurrent layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bias=True)

        # output layer
        self.value_fc = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True)
        )
        self.value_fc[-1].weight.data.uniform_(-init_w, init_w)

    def init_hidden_states(self, bsize):
        hx = torch.zeros(1, bsize, self.hidden_size).float().to(d)
        cx = torch.zeros(1, bsize, self.hidden_size).float().to(d)
        return hx, cx

    def forward(self, state, action_v, param, hidden_in, padding=True, seq_dim_len=None):
        if padding:
            batch_size = len(state)
            state_padded, _ = padding_and_convert(state, self.max_steps)

            cat_tensor = torch.cat((state_padded, action_v, param), 2)
            hidden = self.net(cat_tensor)
            hidden = hidden.view(batch_size, -1, self.hidden_size)

            hidden_packed = pack_padded_sequence(
                hidden, seq_dim_len, enforce_sorted=False, batch_first=True)

            lstm_out = self.lstm(hidden_packed, hidden_in)

            out_packed = lstm_out[0]
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=self.max_steps)
        else:
            batch_size = state.shape[0]
            state = state.view(batch_size, -1, self.num_inputs)
            action_v = action_v.view(batch_size, -1, self.num_d_actions)
            param = param.view(batch_size, -1, self.num_c_actions)

            cat_tensor = torch.cat((state, action_v, param), 2)

            hidden = self.net(cat_tensor)
            hidden = hidden.view(batch_size, -1, self.hidden_size)

            lstm_out = self.lstm(hidden, hidden_in)
            out = lstm_out[0]

        hx = lstm_out[1][0]
        cx = lstm_out[1][1]

        out = self.value_fc(out)
        return out, (hx, cx)


class PASAC_PolicyNetwork_MLP(nn.Module):
    def __init__(self, num_inputs, max_steps, num_d_actions, num_c_actions, hidden_size, action_range=1.,
                 batch_size=512,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        """
        :param num_inputs: state dim
        :param num_actions: discrete action dim
        :param num_params: continuous action dim
        """
        super(PASAC_PolicyNetwork_MLP, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range
        self.max_steps = max_steps
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )

        # output layer
        self.action_fc = nn.Sequential(
            nn.Linear(hidden_size, num_d_actions, bias=True)
        )
        # normalize over action dim / add temperature
        self.action_fc_norm = TemperatureSoftmax(temperature=1., dim=1)
        self.action_fc[-1].weight.data.uniform_(-init_w, init_w)

        # init output layer for PASAC
        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_size, num_c_actions, bias=True))
        self.mean_linear[-1].weight.data.uniform_(-init_w, init_w)
        self.mean_linear[-1].bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Sequential(
            nn.Linear(hidden_size, num_c_actions, bias=True))
        self.log_std_linear[-1].weight.data.uniform_(-init_w, init_w)
        self.log_std_linear[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        out = self.net(state)

        d_action_fc = self.action_fc(out)
        d_action_prob = self.action_fc_norm(d_action_fc)

        mean = self.mean_linear(out)
        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std, d_action_prob

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, d_action_prob = self.forward(state)

        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(d)
        c_action_0 = torch.tanh(mean + std * z)
        c_action = self.action_range * c_action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(
            1. - c_action_0.pow(2) + epsilon) - np.log(self.action_range)

        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return d_action_prob, c_action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(d)
        mean, log_std, d_action_prob = self.forward(state)

        mean = mean[0]  # unsqueeze
        log_std = log_std[0]

        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample().to(d)

        c_action = self.action_range * torch.tanh(mean + std * z)
        c_action = (self.action_range * torch.tanh(mean).detach().cpu().numpy()
                    if deterministic else c_action.detach().cpu().numpy())

        return c_action, d_action_prob[0].detach().cpu().numpy()


class PASAC_QNetwork_MLP(nn.Module):
    def __init__(self, max_steps, num_inputs, num_d_actions, num_c_actions, hidden_size, batch_size=512, init_w=3e-3):
        """
        :param num_inputs: state dim
        :param num_actions: discrete action dim
        :param num_params: continuous action dim
        """
        super(PASAC_QNetwork_MLP, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.num_d_actions = num_d_actions
        self.num_c_actions = num_c_actions

        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_d_actions + num_c_actions, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )

        # output layer
        self.value_fc = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=True)
        )

        self.value_fc[-1].weight.data.uniform_(-init_w, init_w)
        self.value_fc[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action_v, param):
        cat_tensor = torch.cat((state, action_v, param), 1)
        hidden = self.net(cat_tensor)
        out = self.value_fc(hidden)
        return out
