import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.image_utils import to_grayscale, crop, normalize
from torch.optim import Adam
import numpy as np
from torch.distributions import Categorical
from collections import deque
from skimage.transform import resize
from VAE import VAE
from MDNRNN import MDNRNN
from params import Params

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]

def get_action_space():  # action 수 반환
    return len(ACTIONS)

def get_action(probs, train=False, step=None, params=None, device=None):
    #print(probs)
    ## sampling??
    probs = probs.detach()
    probs = Categorical(probs)
    action_index = probs.sample().item()
    #action_index = probs.argmax()
    action = ACTIONS[action_index]
    return [action_index], action

class A2C(nn.Module):
    def __init__(self, num_of_actions):
        super().__init__()
        self.data = []
        self.policy = nn.Linear(256+32, num_of_actions)
        self.value = nn.Linear(256+32, 1)

    def pi(self, x):
        prob = F.softmax(self.policy(x))
        return prob

    def v(self, x):
        value_output = self.value(x)
        return value_output

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 1.0 if done else 0.0
            done_lst.append(done_mask)

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = s_lst, a_lst, r_lst, s_prime_lst, done_lst

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, VAE, MDNRNN, skip_steps):
        super().__init__(env)
        self.VAE = VAE
        self.MDNRNN = MDNRNN
        self.skip_steps = skip_steps

    def reset(self):                                                                           # env.reset을 전처리하여 반환
        state = self.env.reset()
        preprocessed_state = self.preprocess(state)
        return preprocessed_state

    def step(self, action):
        total_reward = 0
        for i in range(self.skip_steps):
            state, reward, done, _ = self.env.step(action)  # 선택한 action으로 진행             # 보여주기
            #self.env.render()
            total_reward += reward  # reward 계산
            if done:
                break
        preprocessed_state = self.preprocess(state)                                            # action 진행 후 state 전처리
        return preprocessed_state, reward, done

    def preprocess(self, state):                                                               # 전처리
        preprocessed_state = self._process_frame(state)
        preprocessed_state = torch.tensor([preprocessed_state], dtype=torch.float32)
        preprocessed_state = np.transpose(preprocessed_state, (0, 3, 1, 2))
        # VAE : state -> z
        z, _, _ = self.VAE.encode(preprocessed_state)
        z = z.view(1, 1, -1)
        # MDNRNN : z -> h -> z'
        (pi, mu, sigma), (hidden, cell) = self.MDNRNN(z)
        # state = z + h
        self.hidden = (hidden, cell)
        preprocessed_state = torch.cat([z, hidden], dim=2)
        preprocessed_state = preprocessed_state.view(-1)
        return preprocessed_state

    def _process_frame(self, frame):
        obs = frame[0:84, :, :].astype(np.float) / 255.0
        obs = resize(obs, (64, 64))
        return obs

def evaluate_a2c(path):
    vae = VAE()
    mdnrnn = MDNRNN()
    vae.eval()
    mdnrnn.eval()
    vae.load_state_dict(torch.load('models/vae.pt'))
    mdnrnn.load_state_dict(torch.load('models/mdnrnn.pt'))
    model = A2C(num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    env_wrapper = EnvironmentWrapper(env, vae, mdnrnn, 1)  # EnvironmentWrapper 생성

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):  # episode 수 만큼 반복
        state = env_wrapper.reset()  # env reset
        state = torch.tensor(state, dtype=torch.float)
        done = False
        score = 0
        while not done:  # 완료될 때까지 반복
            probs = model.pi(torch.stack([state]))
            action_index, action = get_action(probs, train=False)  # 현재 q-value에서의 action 찾고 step
            print(action)
            state, reward, done = env_wrapper.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            score += reward  # score 계산
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score  # 끝났다면 total reward 계산
    return total_reward / num_of_episodes  # episode 평균 reward 반환

def a2c_inference(path):
    vae = VAE()
    mdnrnn = MDNRNN()
    vae.eval()
    mdnrnn.eval()
    vae.load_state_dict(torch.load('models/vae.pt'))
    mdnrnn.load_state_dict(torch.load('models/mdnrnn.pt'))
    model = A2C( num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    env_wrapper = EnvironmentWrapper(env, vae, mdnrnn, 1)  # EnvironmentWrapper 생성

    state = env_wrapper.reset()  # env reset
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_score = 0
    while not done:  # 완료될 때 까지
        probs = model.pi(torch.stack([state]))
        action_index, action = get_action(probs, train=False)  # 현재 q-value에서의 action 찾고 step
        print(action)
        state, reward, done = env_wrapper.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        total_score += reward  # reward 계산
        env_wrapper.render()
    return total_score

class WorldA2CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = A2C(num_of_actions=get_action_space())
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.params.lr)
        self.VAE = VAE()
        self.MDNRNN = MDNRNN()
        env = gym.make('CarRacing-v0')
        self.environment = EnvironmentWrapper(env, self.VAE, self.MDNRNN, 4)

    def run(self):
        self.VAE.load_state_dict(torch.load('models/vae.pt'))
        self.MDNRNN.load_state_dict(torch.load('models/mdnrnn.pt'))
        self.VAE.eval()
        self.MDNRNN.eval()
        num_of_steps = self.params.num_of_steps
        state = torch.tensor(self.environment.reset(),device=self.device,dtype=torch.float32)
        score = 0
        probs_expected = 0
        temp_step = 0
        for step in range(num_of_steps):
            probs = self.model.pi(torch.stack([state]))
            action_index, action = get_action(probs=probs, train=True, step=step, params=self.params, device=self.device)
            next_state, reward, done = self.environment.step(action)
            next_state = torch.tensor(next_state,
                                      device=self.device,
                                      dtype=torch.float32)
            self.model.put_data((state, action_index, reward, next_state, done))
            state = next_state
            score += reward
            probs_expected += probs
            temp_step += 1
            if done:
                state = torch.tensor(self.environment.reset(), device=self.device, dtype=torch.float32)
                print('Step: {0} Score: {1:.2f}'.format(step, score))
                print(probs_expected/temp_step)
                score = 0
                probs_expected = 0
                temp_step = 0
                self.train_net()  # loss 계산
            torch.save(self.model.state_dict(), self.model_path)
        self.environment.close()


    def train_net(self):
        batch = self.model.make_batch()  # 메모리에서 랜덤하게 선택
        states, actions, rewards, next_states, dones = batch
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, device=self.device).view(-1, 1)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).view(-1, 1)
        td_target = rewards + self.params.discount_factor * (self.model.v(next_states) * (1 - dones))
        delta = td_target - self.model.v(states)
        pi = self.model.pi(states)
        policy_a = pi.gather(1, actions)
        loss = -torch.log(policy_a) * (delta.detach()) + F.smooth_l1_loss(self.model.v(states), td_target.detach())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


if __name__ == "__main__":
    params = Params('params/' + "dqn" + '.json')
    model_path = ('models/a2c_world.pt')
    #WorldA2CTrainer = WorldA2CTrainer(params, model_path)
    #WorldA2CTrainer.run()
    a2c_inference(model_path)
    #evaluate_a2c(model_path)