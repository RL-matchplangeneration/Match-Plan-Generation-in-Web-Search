import gym

import itertools
import numpy as np

from Benchmarks.common.wrappers import ScaledStateWrapper


def merge_list(lists):
    return tuple(itertools.chain(*lists))


def wrap_list(list_old, lens):
    res, i = [], 0
    for l in lens:
        res.append(list_old[i:i + l])
        i += l
    return res


class ActionUnwrap(gym.ActionWrapper):
    """ totally flatten -- remove correspondence in the action spaces of the benchmarks """

    def __init__(self, env):
        super(ActionUnwrap, self).__init__(env)
        spaces = env.action_space.spaces
        self.wrapped_discrete = not isinstance(spaces[1], gym.spaces.Tuple)  # Box
        if not self.wrapped_discrete:  # old gym's Tuple needs extra `.spaces`
            num = len(spaces[1].spaces)  # spaces[0].n / len(spaces[1]) (!= n in Soccer-v0)
            high = np.array(merge_list([spaces[1].spaces[i].high for i in range(num)]))
            low = np.array(merge_list([spaces[1].spaces[i].low for i in range(num)]))
            self.lens = [spaces[1].spaces[i].shape[0] for i in range(num)]
        else:  # Soccer-v0 -- discrete action spaces not wrapper by a Tuple
            num = len(spaces) - 1  # first entry is Discrete action / idx from 1
            high = np.array(merge_list([spaces[i + 1].high for i in range(num)]))
            low = np.array(merge_list([spaces[i + 1].low for i in range(num)]))
            self.lens = [spaces[i + 1].shape[0] for i in range(num)]  # TODO
        self.action_space = gym.spaces.Tuple((spaces[0], gym.spaces.Box(low=low, high=high)))

    def action(self, action):
        assert isinstance(action, tuple)
        param = wrap_list(action[1], self.lens)
        return (np.concatenate((np.array([action[0]]), param))
                if self.wrapped_discrete else action[0], param)  # TODO

    def reverse_action(self, action):
        raise NotImplementedError


class StateUnwrap(gym.ObservationWrapper):
    """ undo the "compound" operation in the benchmark """

    def __init__(self, env):
        super(StateUnwrap, self).__init__(env)
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = env.observation_space
        elif isinstance(env.observation_space, gym.spaces.Tuple):
            self.observation_space = env.observation_space.spaces[0]
        else:
            raise Exception('Not Support')

    def observation(self, observation):
        return observation[0] if isinstance(observation, tuple) else observation
