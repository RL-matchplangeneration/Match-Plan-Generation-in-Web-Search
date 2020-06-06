import click
import os
import warnings
from Benchmarks.experiment_benchmark import train_nni
import torch


@click.command()
@click.option('--env_name', default='Platform-v0')  # = Platform-v0 / Goal-v0
@click.option('--seed', default=0)
@click.option('--log_name', default="benchmark_nni")
@click.option('--weights', default=[1., -1, 0.])  # [reward design]
@click.option('--gamma', default=0.95)
@click.option('--replay_buffer_size', default=50000)
# [network]
@click.option('--hidden_size', default=128)
@click.option('--value_lr', default=3e-4)
@click.option('--policy_lr', default=3e-4)
# [training]
@click.option('--train_episodes', default=500000)
@click.option('--batch_size', default=256)
@click.option('--update_freq', default=1)
@click.option('--eval_freq', default=10)
@click.option('--use_exp', default=True)  # True = exp, False = Sample
@click.option('--soft_tau', default=1e-2)
# [test param]
@click.option('--tensorboard_dir', default="./tensorboard_log", type=str)
@click.option('--rnn_step', default=10)  # only for benchmarks with long horizon
def run(env_name,
        seed,
        weights,
        gamma,
        replay_buffer_size,
        hidden_size,
        value_lr,
        policy_lr,
        max_steps,
        train_episodes,
        batch_size,
        update_freq,
        eval_freq,
        use_exp,
        soft_tau,
        log_name,
        tensorboard_dir,
        rnn_step
        ):
    print('\n'.join(['%s:%s' % item for item in locals().items()]))

    debug = {
        'log': log_name,
        'tensorboard_dir': tensorboard_dir,
        'debug': True,

        'print_step': False,
        'print_freq': 32,  # print log in update per 'print_freq' episodes

        'save_model': True,  # save checkpoints or not
        'load_model': False,  # load checkpoints at the beginning or not, 'load_filename' should be assigned
        'checkpoint_save_path': './cp',
        'load_filename': None,  # prefix of the checkpoint
        'save_freq': 5000,  # save a model per 'save_freq' episodes

        'sampling': True,
        'alpha_lr': 3e-4,  # learning rate for tuning PASAC alphas
        'L2_norm': 0,
        'replay_buffer': 'p', # r for normal ER, p for SPER

        'rnn_step': rnn_step
    }

    print('=================debug parameters=================')
    print('\n'.join(['%s:%s' % item for item in debug.items()]))
    print('=================debug parameters=================')

    warnings.filterwarnings("ignore")

    params = locals()  # get local parameters, including all arguments
    if 'Platform' in env_name:
        max_steps = 250
    elif 'Goal' in env_name:
        max_steps = 150
    else:
        raise NotImplementedError

    params.update({'max_steps': max_steps})
    del params['tensorboard_dir']
    del params['log_name']
    print('[params]', params)

    seed_torch(seed)
    train_nni(**params, use_nni=True)


def seed_torch(seed=1029):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('******PID:' + str(os.getpid()) + '******')
    run()
    print('============Done============')
