import numpy as np
import gym
from env import SimpleExplore


ACTIONS = {
    'forward': dict(forward=np.array(1), jump=np.array(1), camera=np.array([0., 0.])),
    'left': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., -20.])),
    'right': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., 20.]))
}

ACTIONS_TO_ID = {
    'forward': 0,
    'left': 1,
    'right': 2
}


def sample_action(prob_forward):
    prob_turn = (1 - prob_forward) / 2
    i = np.random.choice(['forward', 'left', 'right'],
                         p=[prob_forward, prob_turn, prob_turn])
    return ACTIONS[i]


def main():
    abs_env = SimpleExplore(resolution=(128, 128), biomes=[6])# biomes=[140, 38, 158, 133, 4, 27, 134, 8, 37, 165, 38, 166, 13, 17, 18, 19, 31])
    abs_env.register()

    env = gym.make('SimpleExplore-v0')
    for _ in range(100):
        env.reset()
        for _ in range(50000000000000000000000000):
            env.render()
            action = sample_action(0.9)
            env.step(action)


if __name__ == '__main__':
    main()
