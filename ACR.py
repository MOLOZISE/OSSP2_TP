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

# Battlezone https://github.com/mgbellemare/Arcade-Learning-Environment

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]

def get_action_space():
    return len(ACTIONS)

def get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions

# openai gym의 wrapper 활용?
class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self):
        state = self.env.reset()
        for _ in range(self.stack_size):
            self.frames.append(self.preprocess(state))
        return self.state()

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.env.env.viewer.window.dispatch_events()
        preprocessed_state = self.preprocess(state)

        self.frames.append(preprocessed_state)
        return self.state(), reward, done

    def state(self):
        return np.stack(self.frames, axis=0)

    def preprocess(self, state):
        preprocessed_state = to_grayscale(state)
        preprocessed_state = zero_center(preprocessed_state)
        preprocessed_state = crop(preprocessed_state)
        return preprocessed_state

    def get_state_shape(self):
        return (self.stack_size, 84, 84)

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
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32*9*9, 256)
        self.policy = nn.Linear(256, num_of_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output

def make_environment(stack_size):
    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, stack_size)
    return env_wrapper

# 동기적 Actor Critic
def worker(connection, stack_size):
    env = make_environment(stack_size)
    ####
    env.reset()
    ####

    while True:
        command, data = connection.recv()
        if command == 'step':
            state, reward, done = env.step(data)
            if done:
                state = env.reset()
            connection.send((state, reward, done))
        elif command == 'reset':
            state = env.reset()
            connection.send(state)

class ParallelEnvironments:
    def __init__(self, stack_size, number_of_processes=multiprocessing.cpu_count()):
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
        return (self.stack_size, 84, 84)

class A2CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = multiprocessing.cpu_count()
        self.parallel_environments = ParallelEnvironments(self.params.stack_size,
                                                          number_of_processes=self.num_of_processes)
        self.actor_critic = ActorCritic(self.params.stack_size, get_action_space()) # 5, 4
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.params.lr)
        self.storage = A2CStorage(self.params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        print(self.current_observations.size()) # 16 5 84 84

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            for step in range(self.params.steps_per_update):
                probs, log_probs, value = self.actor_critic(self.current_observations)
                actions = get_actions(probs)
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                states, rewards, dones = self.parallel_environments.step(actions)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values,
                                                                     self.params.discount_factor)
            advantages = torch.tensor(expected_rewards) - self.storage.values
            value_loss = advantages.pow(2).mean()
            policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.params.max_norm)
            self.optimizer.step()

            if update % 300 == 0:
                torch.save(self.actor_critic.state_dict(), self.model_path)

            if update % 100 == 0:
                print('Update: {}. Loss: {}'.format(update, loss))

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)

        entropies = -(log_probs * probs).sum(-1)

        return action_log_probs, entropies

# 비동기적 Actor Critic
# 각 Process별로 run
class Worker(mp.Process):
    def __init__(self, process_num, global_model, params):
        super().__init__()
        # Worker 변수
        self.process_num = process_num
        self.global_model = global_model
        # Model 변수
        self.params = params
        # # Environment
        # env = gym.make('CarRacing-v0')
        # ####
        # #env.reset()
        # ####
        # # EnvironmentWrapper
        # self.environment = EnvironmentWrapper(env, self.params.stack_size)
        # ####
        # #self.environment.env.reset()
        # ####
        # AC 모델
        self.model = ActorCritic(self.params.stack_size, get_action_space())
        # Optimizer
        self.optimizer = Adam(self.global_model.parameters(), lr=self.params.lr)
        # 기억 저장 메모리
        self.storage = A3CStorage(self.params.steps_per_update)
        # 현재 observation
        #self.current_observation = torch.zeros(1, *self.environment.get_state_shape())

    def run(self):
        # Environment
        env = gym.make('CarRacing-v0')
        ####
        # env.reset()
        ####
        # EnvironmentWrapper
        self.environment = EnvironmentWrapper(env, self.params.stack_size)
        ####
        # self.environment.env.reset()
        ####
        # update 단위 횟수 (rollout)
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
                action = get_actions(probs)[0]
                action_log_prob, entropy = self.compute_action_log_and_entropy(probs, log_probs)
                # environment Wrapper
                state, reward, done = self.environment.step(action)
                if done:
                    state = self.environment.reset()
                # on torch
                done = torch.Tensor([done])
                self.current_observation = torch.Tensor([state])
                # storage save
                self.storage.add(step, value, reward, action_log_prob, entropy, done)

            # rollout 횟수 종료 후
            # last_value 값 얻기
            _, _, last_value = self.model(self.current_observation)
            # 각 경험에서의 expected_reward가 계산
            expected_reward = self.storage.compute_expected_reward(last_value,
                                                                   self.params.discount_factor)
            # A = TD Target - Value
            advantages = torch.tensor(expected_reward) - self.storage.values
            value_loss = advantages.pow(2).mean()
            if self.params.use_gae:
                gae = self.storage.compute_gae(last_value,
                                               self.params.discount_factor,
                                               self.params.gae_coef)
                policy_loss = -(torch.tensor(gae) * self.storage.action_log_probs).mean()
            else:
                policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            # policy_loss, maximum entropy Inverse reinforcement Leaning, value loss
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward()
            # cliping grad
            nn.utils.clip_grad_norm(self.model.parameters(), self.params.max_norm)
            self._share_gradients()
            self.optimizer.step()

            if update % 20 == 0:
                print('Process: {}. Update: {}. Loss: {}'.format(self.process_num,
                                                                 update,
                                                                 loss))

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
        self.num_of_processes = mp.cpu_count()
        self.global_model = ActorCritic(self.params.stack_size,
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
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    total_reward = 0
    num_of_episodes = 100

    for episode in range(num_of_episodes):
        state = env_wrapper.reset()
        state = torch.Tensor([state])
        done = False
        score = 0
        while not done:
            probs, _, _ = model(state)
            action = get_actions(probs)
            state, reward, done = env_wrapper.step(action[0])
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
    env_wrapper = EnvironmentWrapper(env, params.stack_size)

    state = env_wrapper.reset()
    state = torch.Tensor([state])
    done = False
    total_score = 0
    while not done:
        probs, _, _ = model(state)
        action = get_actions(probs)
        print(action)
        state, reward, done = env_wrapper.step(action[0])
        state = torch.Tensor([state])
        total_score += reward
        env_wrapper.render()
    return total_score

