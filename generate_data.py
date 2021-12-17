# xvfb-run -s "-screen 0 1400x900x24" python 01_generate_data.py car_racing --total_episodes 4000 --time_steps 300

import numpy as np
import random
#import matplotlib.pyplot as plt
#from DQN_ import EnvironmentWrapper
import gym
from skimage.transform import resize
import argparse

DIR_NAME = 'data/rollout_worldmodel/'

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]

class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)                                                          # k frame skip 설정

    def reset(self):                                                                           # env.reset을 전처리하여 반환
        state = self.env.reset()
        preprocessed_state = self.preprocess(state)
        return [preprocessed_state]

    def step(self, action):
        total_reward = 0
        for i in range(4):
            state, reward, done, _ = self.env.step(action)  # 선택한 action으로 진행
            total_reward += reward  # reward 계산
            if done:
                break
        preprocessed_state = self.preprocess(state)                                            # action 진행 후 state 전처리
        return [preprocessed_state], reward, done

    def preprocess(self, state):
        preprocessed_state = self._process_frame(state)
        return preprocessed_state

    def _process_frame(self, frame):
        obs = frame[0:84, :, :].astype(np.float) / 255.0
        obs = resize(obs, (64, 64))
        return obs

def main(args):
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    env = gym.make('CarRacing-v0')
    env = EnvironmentWrapper(env)
    s = 0
    while s < total_episodes:
        episode_id = random.randint(0, 2 ** 31 - 1)
        filename = DIR_NAME + str(episode_id) + ".npz"
        observation = env.reset()
        t = 0
        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        done_sequence = []
        reward = -0.1
        done = False
        while t < time_steps:
            action_index = random.randrange(len(ACTIONS))
            action = ACTIONS[action_index]
            obs_sequence.append(observation)
            action_sequence.append(action)
            reward_sequence.append(reward)
            done_sequence.append(done)
            observation, reward, done = env.step(action)
            t = t + 1
            if done:
                break
        print("Episode {} finished after {} timesteps".format(s, t))
        np.savez_compressed(filename, obs=obs_sequence, action=action_sequence,
                            reward=reward_sequence, done=done_sequence)
        s = s + 1
    env.close()

# python 01_generate_data.py $1 --total_episodes $3 --time_steps $4 --render $5 --action_refresh_rate $6 &

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--total_episodes', type=int, default=10000,
                        help='total number of episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=100,
                        help='how many timesteps at start of episode?')
    args = parser.parse_args()
    main(args)