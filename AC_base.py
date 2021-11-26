import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#hyperparameters
learning_rate = 0.0002
gamma = 0.98
#hyperparameter for qac
n_rollout = 10
# TD AC
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  # input_shape -> 16 -> 32 -> 32
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32 * 7 * 7, 256)  # 32*7*7 -> 256 -> num_of_actions
        self.policy = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):  # 정책 함수
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)

        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        return policy_output

    def v(self, x):  # 가치 함수
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)

        linear1_out = self.linear1(flattened)
        value_output = self.value(linear1_out)
        return value_output

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        # policy update(v 대신 a의 불편추정량 delta 사용) + value update
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def TDmain():
  env = gym.make('CarRacing-v0')
  model = ActorCritic()
  print_interval = 20
  score = 0.0

  for n_epi in range(5000):
    done = False
    s = env.reset()

    while not done:
      for t in range(n_rollout):
        prob = model.pi(torch.from_numpy(s))
        m = Categorical(prob)
        a = m.sample().item()
        s_prime, r, done, info = env.step(a)
        model.put_data((s, a, r,s_prime, done))

        s = s_prime
        score += r

        if done:
          break

      model.train_net()

    if n_epi%print_interval==0 and n_epi!= 0:
      print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
      score = 0.0

  env.close()

if __name__ == '__main__':
    TDmain()