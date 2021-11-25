import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
from torch.optim import RMSprop

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

class DQN(nn.Module):  # DQN 모델
    def __init__(self, input_shape=3, num_of_actions=4):
        super().__init__()
        # input_channel_size(RGB...), output_volume_size(Arbitary), filter_size, padding, stride
        # output size = (input_volume_size - kernel_size + 2 * padding_size) / strides + 1
        # input_volume_size = image width
        # maxpooling 2 -> input_filter_size / 2 = output_filter_size

        # L1 ImgIn shape = (?, 96, 96, 3)
        # conv           = (?, 96, 96, 32)
        # pooling        = (?, 48, 48, 32)
        self.convlayer1 = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L2 ImgIn shape = (?, 48, 48, 32)
        # conv           = (?, 48, 48, 64)
        # pooling        = (?, 24, 24, 64)
        self.convlayer2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L3 ImgIn shape = (?, 24, 24, 64)
        # conv           = (?, 24, 24, 128)
        # pooling        = (?, 12, 12, 128)
        self.convlayer3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(12 * 12 * 128, 4096, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(4096, 1024, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.value = nn.Linear(1024, 1, bias=True)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        conv1_out = self.convlayer1(x)
        conv2_out = self.convlayer2(conv1_out)
        conv3_out = self.convlayer3(conv2_out)
        flattened = conv3_out.view(conv3_out.size(0), -1)
        linear1_out = self.fc1(flattened)
        linear2_out = self.fc2(linear1_out)
        value_output = self.value(linear2_out)

        return value_output

def evaluate_dqn(path):
    model = DQN(input_shape=1, num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    env_wrapper = env  # EnvironmentWrapper 생성

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):  # episode 수 만큼 반복
        state = env_wrapper.reset()  # env reset
        state = torch.tensor([state], dtype=torch.float32)
        done = False
        score = 0
        while not done:  # 완료될 때까지 반복
            q_value = model(state)
            _, action = get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
            #print(action)
            state, reward, done, _ = env_wrapper.step(action)
            state = torch.tensor([state], dtype=torch.float32)
            score += reward  # score 계산
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score  # 끝났다면 total reward 계산
    return total_reward / num_of_episodes  # episode 평균 reward 반환

def dqn_inference(path):
    model = DQN(input_shape=1, num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정
    env_wrapper = env  # EnvironmentWrapper 생성

    state = env_wrapper.reset()  # env reset
    state = torch.tensor([state], dtype=torch.float32)
    done = False
    total_score = 0
    while not done:  # 완료될 때 까지
        q_value = model(state)
        _, action = get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
        #print(action)
        state, reward, done, _ = env_wrapper.step(action)
        state = torch.tensor([state], dtype=torch.float32)
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

class DQNTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_q_net = DQN(input_shape=1, num_of_actions=get_action_space())
        self.current_q_net.to(self.device)
        self.target_q_net = DQN(input_shape=1, num_of_actions=get_action_space())
        self.target_q_net.to(self.device)
        self.optimizer = RMSprop(self.current_q_net.parameters(),
                                 lr=self.params.lr)
        self.replay_memory = ReplayMemory(self.params.memory_capacity)

    def run(self):
        env = gym.make('CarRacing-v0')
        self.environment = env
        state = self.environment.reset()
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        self._update_target_q_net()  # target network 갱신
        for step in range(int(self.params.num_of_steps)):
            q_value = self.current_q_net(state)
            action_index, action = get_action(q_value,  # action 선택 후 진행
                                              train=True,
                                              step=step,
                                              params=self.params,
                                              device=self.device)
            next_state, reward, done, _ = self.environment.step(action)
            next_state = torch.tensor([next_state], device=self.device,
                                     dtype=torch.float32)
            self.replay_memory.add(state, action_index, reward, next_state, done)
            state = next_state
            if done:
                state = self.environment.reset()
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
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
        states = states.view(-1, 96, 96, 3)
        next_states = torch.stack(next_states)
        next_states = next_states.view(-1, 96, 96, 3)
        actions = torch.stack(actions).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.current_q_net(states).max(1)[0]  # q(s, a)
        next_q_values = self.target_q_net(next_states).max(1)[0]  # q'(s', a')

        expected_q_values = rewards + self.params.discount_factor * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()  # Current Q net W Update
        self.optimizer.step()
        return loss

    def _update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())  # target network를 current q network로 갱신s

