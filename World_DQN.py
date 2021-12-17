import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils.image_utils import to_grayscale, crop, normalize
from collections import deque
from torch.optim import RMSprop
from VAE import VAE
from MDNRNN import MDNRNN
import numpy as np
from params import Params
from skimage.transform import resize

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():  # action 수 반환
    return len(ACTIONS)

def get_action(q_value, train=False, step=None, params=None, device=None):
    if train:  # train의 경우
        epsilon = params.epsilon_final + (params.epsilon_start - params.epsilon_final) * \
                  math.exp(-1 * step / params.epsilon_step)  # epsilon greedy에 의해 action 선택
        if random.random() <= epsilon:
            action_index = random.randrange(get_action_space())
            action = ACTIONS[action_index]
            return torch.tensor([action_index], device=device)[0], action
    action_index = q_value.max(1)[1]  # train이 아닌 경우 q밸류에 의해 action 선택
    action = ACTIONS[action_index[0]]
    return action_index[0], action

class Con(nn.Module):  # DQN 모델
    def __init__(self, num_of_actions):
        super().__init__()
        self.linear1 = nn.Linear(256+32, num_of_actions)

    def forward(self, x):
        q_value = (self.linear1(x))
        return q_value

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

def evaluate_dqn(path):
    vae = VAE()
    mdnrnn = MDNRNN()
    vae.eval()
    mdnrnn.eval()
    vae.load_state_dict(torch.load('models/vae.pt'))
    mdnrnn.load_state_dict(torch.load('models/mdnrnn.pt'))
    model = Con(num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    #env_wrapper = EnvironmentWrapper(env, 1)  # EnvironmentWrapper 생성
    env_wrapper = EnvironmentWrapper(env, vae, mdnrnn, 1)

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):  # episode 수 만큼 반복
        state = env_wrapper.reset()  # env reset
        state = torch.tensor(state, dtype=torch.float)
        done = False
        score = 0
        while not done:  # 완료될 때까지 반복
            q_value = model(torch.stack([state]))
            _, action = get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
            #print(action)
            state, reward, done = env_wrapper.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            score += reward  # score 계산
            #env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score  # 끝났다면 total reward 계산
    return total_reward / num_of_episodes  # episode 평균 reward 반환

def dqn_inference(path):
    vae = VAE()
    mdnrnn = MDNRNN()
    vae.eval()
    mdnrnn.eval()
    vae.load_state_dict(torch.load('models/vae.pt'))
    mdnrnn.load_state_dict(torch.load('models/mdnrnn.pt'))
    model = Con(num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    #env_wrapper = EnvironmentWrapper(env, 1)  # EnvironmentWrapper 생성
    env_wrapper = EnvironmentWrapper(env, vae, mdnrnn, 1)

    state = env_wrapper.reset()  # env reset
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_score = 0
    while not done:  # 완료될 때 까지
        q_value = model(torch.stack([state]))
        _, action = get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
        print(action)
        state, reward, done = env_wrapper.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        total_score += reward  # reward 계산
        env_wrapper.render()
    return total_score

class ReplayMemory:  # Replay Memory 클래스
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 메모리 저장

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)  # 저장된 메모리 중 랜덤으로 선택
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

class WorldDQNTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_q_net = Con(num_of_actions=get_action_space())
        self.current_q_net.to(self.device)
        self.target_q_net = Con(num_of_actions=get_action_space())
        self.target_q_net.to(self.device)
        self.optimizer = RMSprop(self.current_q_net.parameters(),
                                 lr=self.params.lr)
        self.VAE = VAE()
        self.MDNRNN = MDNRNN()
        self.replay_memory = ReplayMemory(self.params.memory_capacity)
        env = gym.make('CarRacing-v0')
        self.environment = EnvironmentWrapper(env, self.VAE, self.MDNRNN, 4)

    def run(self):
        self.VAE.load_state_dict(torch.load('models/vae.pt'))
        self.MDNRNN.load_state_dict(torch.load('models/mdnrnn.pt'))
        self.VAE.eval()
        self.MDNRNN.eval()
        state = torch.tensor(self.environment.reset(),
                             device=self.device,
                             dtype=torch.float32)
        self._update_target_q_net()  # target network 갱신
        for step in range(int(self.params.num_of_steps)):
            q_value = self.current_q_net(torch.stack([state]))
            action_index, action = get_action(q_value,  # action 선택 후 진행
                                              train=True,
                                              step=step,
                                              params=self.params,
                                              device=self.device)
            next_state, reward, done = self.environment.step(action)
            next_state = torch.tensor(next_state,
                                      device=self.device,
                                      dtype=torch.float32)
            self.replay_memory.add(state, action_index, reward, next_state, done)
            state = next_state
            if done:
                state = torch.tensor(self.environment.reset(),  # 완료되었으면 reset
                                     device=self.device,
                                     dtype=torch.float32)
            if len(self.replay_memory.memory) > self.params.batch_size:  # 메모리가 쌓였다면
                loss = self._update_current_q_net()  # loss 계산
                print('Update: {}. Loss: {}'.format(step, loss))
            if step % self.params.target_update_freq == 0:  # 일정 횟수 step 진행 시 target network 갱신
                self._update_target_q_net()
            torch.save(self.target_q_net.state_dict(), self.model_path)

    def _update_current_q_net(self):
        batch = self.replay_memory.sample(self.params.batch_size)  # 메모리에서 랜덤하게 선택
        states, actions, rewards, next_states, dones = batch

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.current_q_net(states).gather(1, actions)  # q(s, a)
        next_q_values = self.target_q_net(next_states).max(1)[0]  # q'(s', a')

        expected_q_values = rewards + self.params.discount_factor * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()  # Current Q net W Update
        self.optimizer.step()
        return loss

    def _update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())  # target network를 current q network로 갱신

if __name__ == "__main__":
    params = Params('params/' + "dqn" + '.json')
    model_path = ('models/dqn_world.pt')
    #WorldDQNTrainer = WorldDQNTrainer(params, model_path)
    #WorldDQNTrainer.run()
    dqn_inference(model_path)
    #evaluate_dqn(model_path)