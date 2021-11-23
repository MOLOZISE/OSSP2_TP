import multiprocessing
import torch
import torch.nn as nn
from torch.optim import Adam
from A3C_ import ActorCritic
from A3C_ import get_action_space, get_actions
import gym

from multiprocessing import Process, Pipe
from A3C_ import EnvironmentWrapper

# 동기적 Actor Critic

def worker(connection, stack_size):
    env = make_environment(stack_size)                                                              # frame skip size = stack size

    while True:
        command, data = connection.recv()                                                           # 들어온 connection 데이터에 따라 학습을 진행
        if command == 'step':
            state, reward, done = env.step(data)
            if done:
                state = env.reset()
            connection.send((state, reward, done))
        elif command == 'reset':
            state = env.reset()
            connection.send(state)

def make_environment(stack_size):
    env = gym.make('CarRacing-v0')                                                                  # gym environment 설정
    env_wrapper = EnvironmentWrapper(env, stack_size)                                               # frame skip size = stack size인 wrapper 클래스 선언
    return env_wrapper

class ParallelEnvironments:
    def __init__(self, stack_size, number_of_processes=multiprocessing.cpu_count()):                
        self.number_of_processes = number_of_processes                                              # process 개수 = 시스템의 CPU 수 
        self.stack_size = stack_size                                                                # frame skip size = stack size

        # pairs of connections in duplex connection
        self.parents, self.childs = zip(*[Pipe() for _
                                          in range(number_of_processes)])                           # process 개수만큼 Pipe 오브젝트 생성

        self.processes = [Process(target=worker, args=(child, self.stack_size,), daemon=True)       # 각각의 child로 worker 함수를 기반으로 하는 복수의 procecss 생성
                          for child in self.childs]

        for process in self.processes:                                                              # 각 process 실행
            process.start()

    def step(self, actions):
        for action, parent in zip(actions, self.parents):                                           # Process에 step 명령 전달
            parent.send(('step', action))
        results = [parent.recv() for parent in self.parents]                                        # 결과 수신
        states, rewards, dones = zip(*results)
        return torch.Tensor(states), torch.Tensor(rewards), torch.Tensor(dones)                     

    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))                                                            # Process에 reset 명령 전달
        results = [parent.recv() for parent in self.parents]                                        # 결과 수신
        return torch.Tensor(results)

    def get_state_shape(self):
        return (self.stack_size, 84, 84)

class Storage:

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

        for update in range(int(num_of_updates)):                                                                   # Update 수만큼 다음을 반복
            self.storage.reset_storage()                                                                            # Storage reset
            for step in range(self.params.steps_per_update):                                                        # update 당 step만큼 다음을 반복
                probs, log_probs, value = self.actor_critic(self.current_observations)
                actions = get_actions(probs)                                                                        # actor critic에 의해 계산된 값에 따라 action 선택
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                states, rewards, dones = self.parallel_environments.step(actions)                                   # 선택된 action에 따라 진행
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states                  
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)                          # Storage에 저장

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values,
                                                                     self.params.discount_factor)                   # reward 계산
            advantages = torch.tensor(expected_rewards) - self.storage.values
            value_loss = advantages.pow(2).mean()                                                                   # loss 계산
            policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.optimizer.zero_grad()
            loss = policy_loss - self.params.entropy_coef * self.storage.entropies.mean() + \
                self.params.value_loss_coef * value_loss
            loss.backward(retain_graph=True)                                                                        # backpropagate
            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.params.max_norm)
            self.optimizer.step()

            if update % 300 == 0:
                torch.save(self.actor_critic.state_dict(), self.model_path)                                         # update 300번 당 저장

            if update % 100 == 0:
                print('Update: {}. Loss: {}'.format(update, loss))                                                  # update 100번 당 출력

    def compute_action_logs_and_entropies(self, probs, log_probs):                                                  
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)                                                             # action log 계산

        entropies = -(log_probs * probs).sum(-1)                                                                    # entropies 계산

        return action_log_probs, entropies