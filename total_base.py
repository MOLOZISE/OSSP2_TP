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

def q_get_action(q_value, train=False, step=None, params=None, device=None):
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
    model = DQN(num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):  # episode 수 만큼 반복
        state = env.reset()  # env reset
        state = torch.tensor([state], dtype=torch.float32)
        done = False
        score = 0
        while not done:  # 완료될 때까지 반복
            q_value = model(state)
            _, action = q_get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
            #print(action)
            state, reward, done, _ = env.step(action)
            state = torch.tensor([state], dtype=torch.float32)
            score += reward  # score 계산
            env.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score  # 끝났다면 total reward 계산
    return total_reward / num_of_episodes  # episode 평균 reward 반환

def dqn_inference(path):
    model = DQN(input_shape=1, num_of_actions=get_action_space())  # DQN model 생성 후 평가모드 전환
    model.load_state_dict(torch.load(path))  # 학습된 매개변수 불러오기
    model.eval()

    env = gym.make('CarRacing-v0')  # gym environment 설정

    state = env.reset()  # env reset
    state = torch.tensor([state], dtype=torch.float32)
    done = False
    total_score = 0
    while not done:  # 완료될 때 까지
        q_value = model(state)
        _, action = q_get_action(q_value, train=False)  # 현재 q-value에서의 action 찾고 step
        #print(action)
        state, reward, done, _ = env.step(action)
        state = torch.tensor([state], dtype=torch.float32)
        total_score += reward  # reward 계산
        env.render()
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
        self.current_q_net = DQN(num_of_actions=get_action_space())
        self.current_q_net.to(self.device)
        self.target_q_net = DQN(num_of_actions=get_action_space())
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
            action_index, action = q_get_action(q_value,  # action 선택 후 진행
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

import multiprocessing
from multiprocessing import Process, Pipe
import torch
import gym
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import deque
from utils.image_utils import to_grayscale, zero_center, crop
from torch.optim import Adam
import torch.nn.functional as F

def get_action_space():
    return len(ACTIONS)

def ac_get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions

# 경험 저장 메모리
class A2CStorage:

    def __init__(self, steps_per_update, num_of_processes):
        self.steps_per_update = steps_per_update
        self.num_of_processes = num_of_processes
        self.reset_storage()

    def reset_storage(self):                                                                        # Storage 초기화
        self.values = torch.zeros(self.steps_per_update,
                                  self.num_of_processes,
                                  1)
        self.rewards = torch.zeros(self.steps_per_update,
                                   self.num_of_processes,
                                   1)
        self.action_log_probs = torch.zeros(self.steps_per_update,
                                            self.num_of_processes,
                                            1)
        self.entropies = torch.zeros(self.steps_per_update,
                                     self.num_of_processes)
        self.dones = torch.zeros(self.steps_per_update,
                                 self.num_of_processes,
                                 1)

    def add(self, step, values, rewards, action_log_probs, entropies, dones):                      # Storage 추가
        self.values[step] = values
        self.rewards[step] = rewards
        self.action_log_probs[step] = action_log_probs
        self.entropies[step] = entropies
        self.dones[step] = dones

    def compute_expected_rewards(self, last_values, discount_factor):
        expected_rewards = torch.zeros(self.steps_per_update + 1,
                                       self.num_of_processes,
                                       1)
        expected_rewards[-1] = last_values
        for step in reversed(range(self.rewards.size(0))):                                         # reward 계산 후 저장
            expected_rewards[step] = self.rewards[step] + \
                                     expected_rewards[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_rewards[:-1]

# 경험 저장 메모리
class A3CStorage:
    def __init__(self, steps_per_update):
        self.steps_per_update = steps_per_update
        self.reset_storage()

    def reset_storage(self):
        self.values = torch.zeros(self.steps_per_update, 1)
        self.rewards = torch.zeros(self.steps_per_update, 1)
        self.action_log_probs = torch.zeros(self.steps_per_update, 1)
        self.entropies = torch.zeros(self.steps_per_update)
        self.dones = torch.zeros(self.steps_per_update, 1)

    def add(self, step, value, reward, action_log_prob, entropy, done):
        self.values[step] = value
        self.rewards[step] = reward
        self.action_log_probs[step] = action_log_prob
        self.entropies[step] = entropy
        self.dones[step] = done

    # dymanic programming을 위한 역 계산
    def compute_expected_reward(self, last_value, discount_factor):
        expected_reward = torch.zeros(self.steps_per_update + 1, 1)
        expected_reward[-1] = last_value
        for step in reversed(range(self.rewards.size(0))):
            expected_reward[step] = self.rewards[step] + \
                                    expected_reward[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_reward[:-1]

    # general advantage estimation
    def compute_gae(self, last_value, discount_factor, gae_coef):
        gae = torch.zeros(self.steps_per_update + 1, 1)
        next_value = last_value
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + discount_factor * next_value - self.values[step]
            gae[step] = gae[step + 1] * discount_factor * gae_coef + delta
            next_value = self.values[step]
        return gae[:-1]

# Actor Critic building : 생성자, forward 함수 정의
class ActorCritic(nn.Module):
    def __init__(self, num_of_inputs=3, num_of_actions=4):
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
        self.policy = nn.Linear(1024, num_of_actions, bias=True)
        nn.init.xavier_uniform_(self.policy.weight)
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
        policy_output = self.policy(linear2_out)
        value_output = self.value(linear2_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output

# 동기적 Actor Critic
def worker(connection, stack_size):
    env = gym.make('CarRacing-v0')
    env.reset()

    while True:
        command, data = connection.recv()
        #env.render()
        if command == 'step':
            state, reward, done, _ = env.step(data)
            if done:
                state = env.reset()
            connection.send((state, reward, done))
        elif command == 'reset':
            state = env.reset()
            connection.send(state)

class ParallelEnvironments:
    def __init__(self, stack_size, number_of_processes=4):
        self.number_of_processes = number_of_processes
        self.stack_size = stack_size

        # pairs of connections in duplex connection
        self.parents, self.childs = zip(*[Pipe() for _
                                          in range(number_of_processes)])

        self.processes = [Process(target=worker, args=(child, self.stack_size,), daemon=True)
                          for child in self.childs]

        for process in self.processes:
            process.start()

    def step(self, actions):
        for action, parent in zip(actions, self.parents):
            parent.send(('step', action))
        results = [parent.recv() for parent in self.parents]
        states, rewards, dones = zip(*results)
        return torch.Tensor(states), torch.Tensor(rewards), torch.Tensor(dones)

    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))
        results = [parent.recv() for parent in self.parents]
        return torch.Tensor(results)

    def get_state_shape(self):
        #return (self.stack_size, 84, 84)
        return (3, 96, 96)

class A2CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = 4
        self.parallel_environments = ParallelEnvironments(self.params.stack_size,
                                                          number_of_processes=self.num_of_processes)
        self.actor_critic = ActorCritic(3, get_action_space()) # 5, 4
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.params.lr)
        self.storage = A2CStorage(self.params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            for step in range(self.params.steps_per_update):
                probs, log_probs, value = self.actor_critic(self.current_observations)
                actions = ac_get_actions(probs)
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)
                states, rewards, dones = self.parallel_environments.step(actions)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values,
                                                                     self.params.discount_factor)
            advantages = expected_rewards.clone().detach() - self.storage.values
            value_loss = advantages.mean()
            policy_loss = -(advantages * self.storage.action_log_probs).mean()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.params.max_norm)
            self.optimizer.step()

            if update % 20 == 0:
                print('Update: {}. Loss: {}'.format(update, loss))
                torch.save(self.actor_critic.state_dict(), self.model_path)

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)
        entropies = -(log_probs * probs).sum(-1)

        return action_log_probs, entropies

# 비동기적 Actor Critic
# 각 Process별로 run
class Worker(mp.Process):
    def __init__(self, process_num, global_model, params, model_path):
        super().__init__()
        # Worker 변수
        self.process_num = process_num
        self.global_model = global_model
        # Model 변수
        self.params = params
        self.model_path = model_path
        # AC 모델
        self.model = ActorCritic(3, get_action_space())
        # Optimizer
        self.optimizer = Adam(self.global_model.parameters(), lr=self.params.lr)
        # 기억 저장 메모리
        self.storage = A3CStorage(self.params.steps_per_update)

    def run(self):
        # Environment
        env = gym.make('CarRacing-v0')
        self.environment = env
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        # 현재 observation
        self.current_observation = torch.Tensor([self.environment.reset()])

        for update in range(int(num_of_updates)):
            # reset memory
            self.storage.reset_storage()
            # synchronize with global model
            self.model.load_state_dict(self.global_model.state_dict())
            for step in range(self.params.steps_per_update):
                # model은 softmax probs, log_softmax probs, value(estimated)을 forwarding
                probs, log_probs, value = self.model(self.current_observation)
                # softmax된 probs에 대한 action 선택 -> max값
                action = ac_get_actions(probs)[0]
                action_log_prob, entropy = self.compute_action_log_and_entropy(probs, log_probs)
                # environment Wrapper
                state, reward, done, _ = self.environment.step(action)
                if done:
                    state = self.environment.reset()
                # on torch
                done = torch.Tensor([done])
                self.current_observation = torch.Tensor([state])
                # storage save
                self.storage.add(step, value, reward, action_log_prob, entropy, done)

            # last_value 값 얻기
            _, _, last_value = self.model(self.current_observation)
            # 각 경험에서의 expected_reward가 계산
            expected_reward = self.storage.compute_expected_reward(last_value,
                                                                   self.params.discount_factor)
            # A = TD Target - Value
            advantages = torch.tensor(expected_reward.clone().detach()) - self.storage.values
            value_loss = advantages.mean()
            if self.params.use_gae:
                gae = self.storage.compute_gae(last_value,
                                               self.params.discount_factor,
                                               self.params.gae_coef)
                policy_loss = -(torch.tensor(gae) * self.storage.action_log_probs).mean()
            else:
                policy_loss = -(advantages * self.storage.action_log_probs).mean()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            self.optimizer.zero_grad()
            # policy_loss, maximum entropy Inverse reinforcement Leaning, value loss
            loss.backward()
            self._share_gradients()
            self.optimizer.step()

            if update % 20 == 0:
                print('Process: {}. Update: {}. Loss: {}'.format(self.process_num,
                                                                 update,
                                                                 loss))
                torch.save(self.global_model.state_dict(), self.model_path)

    def compute_action_log_and_entropy(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_prob = log_probs.gather(1, indices)

        entropy = -(log_probs * probs).sum(-1)

        return action_log_prob, entropy

    def _share_gradients(self):
        for local_param, global_param in zip(self.model.parameters(),
                                             self.global_model.parameters()):
            global_param._grad = local_param.grad

# A3CTrainer : multiprocessing 사용, Module.sharing memory, 각 Worker를 이용하여 프로세스 생성 및 실행
class A3CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = 4
        self.global_model = ActorCritic(3,
                                        get_action_space())
        self.global_model.share_memory()

    def run(self):
        processes = []
        for process_num in range(self.num_of_processes):
            worker = Worker(process_num, self.global_model, self.params)
            processes.append(worker)
            worker.start()

        for process in processes:
            process.join()

        torch.save(self.global_model.state_dict(), self.model_path)

# AC loading + gym에서 episode -> testing
def evaluate_actor_critic(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = env

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):
        state = env_wrapper.reset()
        state = torch.Tensor([state])
        done = False
        score = 0
        while not done:
            probs, _, _ = model(state)
            action = ac_get_actions(probs)
            state, reward, done, _ = env_wrapper.step(action[0])
            state = torch.Tensor([state])
            score += reward
            env_wrapper.render()
        print('Episode: {0} Score: {1:.2f}'.format(episode, score))
        total_reward += score
    return total_reward / num_of_episodes

# 1번의 episode 진행
def actor_critic_inference(params, path):
    model = ActorCritic(params.stack_size, get_action_space())
    model.load_state_dict(torch.load(path))
    model.eval()

    env = gym.make('CarRacing-v0')
    env_wrapper = env

    state = env_wrapper.reset()
    state = torch.Tensor([state])
    done = False
    total_score = 0
    while not done:
        probs, _, _ = model(state)
        action = ac_get_actions(probs)
        #print(action)
        state, reward, done, _ = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    return total_score

