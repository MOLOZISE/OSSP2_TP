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

LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]

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

class Storage:

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

    def compute_expected_reward(self, last_value, discount_factor):
        expected_reward = torch.zeros(self.steps_per_update + 1, 1)
        expected_reward[-1] = last_value
        for step in reversed(range(self.rewards.size(0))):
            expected_reward[step] = self.rewards[step] + \
                                    expected_reward[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_reward[:-1]

    def compute_gae(self, last_value, discount_factor, gae_coef):
        gae = torch.zeros(self.steps_per_update + 1, 1)
        next_value = last_value
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + discount_factor * next_value - self.values[step]
            gae[step] = gae[step + 1] * discount_factor * gae_coef + delta
            next_value = self.values[step]
        return gae[:-1]

class Worker(mp.Process):
    def __init__(self, process_num, global_model, params):
        super().__init__()

        self.process_num = process_num
        self.global_model = global_model
        self.params = params
        env = gym.make('CarRacing-v0')
        self.environment = EnvironmentWrapper(env, self.params.stack_size)
        self.model = ActorCritic(self.params.stack_size, get_action_space())
        self.optimizer = Adam(self.global_model.parameters(), lr=self.params.lr)
        self.storage = Storage(self.params.steps_per_update)
        self.current_observation = torch.zeros(1, *self.environment.get_state_shape())

    def run(self):
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observation = torch.Tensor([self.environment.reset()])

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            # synchronize with global model
            self.model.load_state_dict(self.global_model.state_dict())
            for step in range(self.params.steps_per_update):
                probs, log_probs, value = self.model(self.current_observation)
                action = get_actions(probs)[0]
                action_log_prob, entropy = self.compute_action_log_and_entropy(probs, log_probs)

                state, reward, done = self.environment.step(action)
                if done:
                    state = self.environment.reset()
                done = torch.Tensor([done])
                self.current_observation = torch.Tensor([state])
                self.storage.add(step, value, reward, action_log_prob, entropy, done)

            _, _, last_value = self.model(self.current_observation)
            expected_reward = self.storage.compute_expected_reward(last_value,
                                                                   self.params.discount_factor)
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
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward()
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

def get_action_space():
    return len(ACTIONS)

def get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions

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

