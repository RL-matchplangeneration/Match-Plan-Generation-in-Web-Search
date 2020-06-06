import torch
from torch.nn import functional as F
import numpy as np
import copy
import datetime
import csv
import codecs
import gym
import os
import warnings
from torch.utils.tensorboard import SummaryWriter

from Experiments_RL.sac_lstm import PASAC_Agent_LSTM
from Experiments_RL.sac_mlp import PASAC_Agent_MLP

from Utils_RL.utils import SPER_MLP

from Benchmarks.utils_benchmark import ActionUnwrap, StateUnwrap
from Benchmarks.common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper, \
    QPAMDPScaledParameterisedActionWrapper
from Benchmarks.common.platform_domain import PlatformFlattenedActionWrapper
from Utils_RL.utils import NormalizedHybridActions, v2id, PATransition

import logging
import nni

logger = logging.getLogger('benchmark_goal_tuning')


def train_nni(**kwargs):
    params = locals()['kwargs']  # get local parameters (in dict kwargs), including all arguments
    if params['use_nni']:
        try:
            # get parameters form tuner
            tuner_params = nni.get_next_parameter()
            print(params, tuner_params)
            params.update(tuner_params)
        except Exception as exception:
            logger.exception(exception)
            raise
        print('[params after NNI]', params)
    train_mlp(**params)


def train_mlp(env_name, debug,
              seed, max_steps, train_episodes,
              batch_size, update_freq, eval_freq,
              weights, gamma, replay_buffer_size,
              hidden_size, value_lr, policy_lr,
              soft_tau=1e-2,
              use_exp=True,
              use_nni=False):
    assert env_name in ['Platform-v0', 'Goal-v0']
    if 'Goal' in env_name:
        import gym_goal
    if 'Platform' in env_name:
        import gym_platform
    env = gym.make(env_name)
    env = ScaledStateWrapper(env)

    env = ActionUnwrap(env)  #  scale to [-1,1] to match the range of tanh
    env = StateUnwrap(env)
    env = NormalizedHybridActions(env)

    # env specific
    state_dim = env.observation_space.shape[0]
    action_discrete_dim, action_continuous_dim = env.action_space.spaces[0].n, env.action_space.spaces[1].shape[0]

    env.seed(seed)
    np.random.seed(seed)

    agent = PASAC_Agent_MLP(debug, weights, gamma, replay_buffer_size, max_steps,
                            hidden_size, value_lr, policy_lr, batch_size, state_dim,
                            action_discrete_dim, action_continuous_dim, soft_tau,
                            use_exp)

    if isinstance(agent.replay_buffer, SPER_MLP):
        agent.replay_buffer.beta_increment_per_sampling = 1. / (max_steps * train_episodes)
    returns = []
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    DIR = debug['tensorboard_dir']
    NAME = os.path.join(DIR, str(start.strftime('%m.%d-%H-%M-%S') + env_name))
    print(NAME)
    if not use_nni:
        writer = SummaryWriter(NAME)

    for episode in range(train_episodes):
        # -----save model-----
        if debug['save_model'] and episode % debug['save_freq'] == 0 and episode > 0:
            print('============================================')
            print("Savepoint - Save model in episodes:", episode)
            print('============================================')
            agent.save(episode)

        # -----reset env-----
        state = env.reset()

        # -----init-----
        episode_reward_sum = 0.

        for step in range(max_steps):
            # -----step-----
            agent.total_step += 1
            action, _, action_v, param = agent.act(state, debug['sampling'])
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, action_v, param, reward, next_state, done)

            # -----move to the next step-----
            state = next_state
            episode_reward_sum += reward

            # -----update models-----
            if len(agent.replay_buffer) > batch_size and step % update_freq == 0:
                agent.update(batch_size,
                             auto_entropy=True,
                             soft_tau=soft_tau,
                             target_entropy=-1. * (action_continuous_dim),
                             need_print=(episode % debug['print_freq'] == 0) and step == 0)

            # -----done-----
            if done:
                break

        if episode % 100 == 0:
            print(f'episode: {episode}, reward: {episode_reward_sum}')
        returns.append(episode_reward_sum)
        if not use_nni:
            writer.add_scalar('Training-Reward-' + env_name, episode_reward_sum, global_step=episode)

        # [periodic evaluation]
        if episode % 10 == 0:  # more frequent
            print(episode, '[time]', datetime.datetime.now() - start,
                  episode_reward_sum, '\n', '>>>>>>>>>>>>>>>>')

            # [evaluation]
            episode_reward_eval = evaluate_mlp(agent, env, max_steps, use_nni, eval_repeat=1)
            if not use_nni:
                writer.add_scalar('EvalReward-' + env_name, episode_reward_eval, global_step=episode)

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    # [report final results]
    average_reward = sum(returns) / len(returns)
    evaluate_mlp(agent, env, max_steps, use_nni, report_avg=average_reward, eval_repeat=100)  # less time

    env.close()
    if not use_nni:
        writer.close()


def evaluate_mlp(agent, env, max_steps, use_nni=False, report_avg=None, eval_repeat=1):
    print("Evaluating agent over {} episodes".format(eval_repeat))
    evaluation_returns = []
    for _ in range(eval_repeat):
        state = env.reset()
        episode_reward = 0.
        for _ in range(max_steps):
            with torch.no_grad():
                action, _, _, _ = agent.act(state, True)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward
            if done:  # currently all situations end with a done
                break

        evaluation_returns.append(episode_reward)
    eval_avg = sum(evaluation_returns) / len(evaluation_returns)
    print("Ave. evaluation return =", eval_avg)

    if use_nni:
        if eval_repeat == 1:
            nni.report_intermediate_result(eval_avg)
        elif eval_repeat > 1 and report_avg is not None:
            metric = (report_avg + eval_avg) / 2
            nni.report_final_result(metric)
    return eval_avg
