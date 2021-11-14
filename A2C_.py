import multiprocessing
import gym
import torch

from multiprocessing import Process, Pipe
from A3C_ import EnvironmentWrapper

def worker(connection, stack_size):
    env = make_environment(stack_size)

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

def make_environment(stack_size):
    env = gym.make('CarRacing-v0')
    env_wrapper = EnvironmentWrapper(env, stack_size)
    return env_wrapper

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

class Storage:

    def __init__(self, steps_per_update, num_of_processes):
        self.steps_per_update = steps_per_update
        self.num_of_processes = num_of_processes
        self.reset_storage()

    def reset_storage(self):
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

    def add(self, step, values, rewards, action_log_probs, entropies, dones):
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
        for step in reversed(range(self.rewards.size(0))):
            expected_rewards[step] = self.rewards[step] + \
                                     expected_rewards[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_rewards[:-1]

import multiprocessing
import torch
import torch.nn as nn
from torch.optim import Adam
from A3C_ import ActorCritic
from A3C_ import get_action_space, get_actions

class A2CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = multiprocessing.cpu_count()
        self.parallel_environments = ParallelEnvironments(self.params.stack_size,
                                                          number_of_processes=self.num_of_processes)
        self.actor_critic = ActorCritic(self.params.stack_size, get_action_space())
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.params.lr)
        self.storage = Storage(self.params.steps_per_update, self.num_of_processes)
        self.current_observations = torch.zeros(self.num_of_processes,
                                                *self.parallel_environments.get_state_shape())

    def run(self):
        # num of updates per environment
        num_of_updates = self.params.num_of_steps / self.params.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        print(self.current_observations.size())

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