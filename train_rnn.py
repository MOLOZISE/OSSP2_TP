# python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

import argparse
import numpy as np
import os
import torch
from torch import optim
from MDNRNN import MDNRNN
from params import Params

ROOT_DIR_NAME = './data/'
SERIES_DIR_NAME = './data/series_worldmodel/'

# latent action hidden gaussian

class MDNRNNTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MDNRNN(z_size=self.params.latent_size)  # learning_rate = LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)

    def get_filelist(self, N):
        filelist = os.listdir(SERIES_DIR_NAME)
        filelist = [x for x in filelist if (x != '.DS_Store' and x != '.gitignore')]
        filelist.sort()
        length_filelist = len(filelist)

        if length_filelist > N:
            filelist = filelist[:N]

        if length_filelist < N:
            N = length_filelist

        return filelist, N

    def random_batch(self, filelist, batch_size):
        N_data = len(filelist)
        indices = np.random.permutation(N_data)[0:batch_size]

        z_list = []
        action_list = []
        rew_list = []
        done_list = []

        for i in indices:
            try:
                new_data = np.load(SERIES_DIR_NAME + filelist[i], allow_pickle=True)
                mu = new_data['mu']
                log_var = new_data['log_var']
                action = new_data['action']
                reward = new_data['reward']
                done = new_data['done']

                reward = np.expand_dims(reward, axis=1)
                done = np.expand_dims(done, axis=1)

                s = log_var.shape
                z = mu + np.exp(log_var / 2.0) * np.random.randn(*s)
                z_list.append(z)
                action_list.append(action)
                rew_list.append(reward)
                done_list.append(done)

            except Exception as e:
                print(e)

        z_list = np.array(z_list)
        action_list = np.array(action_list)
        rew_list = np.array(rew_list)
        done_list = np.array(done_list)

        return z_list, action_list, rew_list, done_list

    def mdn_loss_fn(self, y, pi, mu, sigma):
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss)
        return loss.mean()

    def criterion(self, y, pi, mu, sigma):
        y = y.unsqueeze(2)
        return self.mdn_loss_fn(y, pi, mu, sigma)

    def detach(self, states):
        return [state.detach() for state in states]

    def train_MDNRNN(self, N):
        filelist, N = self.get_filelist(N)
        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        for epoch in range(self.params.epochs):
            for step in range(self.params.steps):
                z, action, rew, done = self.random_batch(filelist, self.params.batch_size)
                # rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis=2)
                # rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :]], axis=2)  # , done[:, 1:, :]
                # if step == 0:
                #     np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input=rnn_input, rnn_output=rnn_output)
                #
                # rnn_in = torch.tensor(rnn_input, dtype=torch.float32)
                # rnn_out = torch.tensor(rnn_output, dtype=torch.float32)
                cur_z = z[:, :-1, :]
                nex_z = z[:, 1:, :]
                rnn_in = torch.tensor(cur_z, dtype=torch.float32)
                rnn_out = torch.tensor(nex_z, dtype=torch.float32)
                (pi, mu, sigma), _ = self.model(rnn_in)
                self.optimizer.zero_grad()
                loss = self.criterion(rnn_out, pi, mu, sigma)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if step % 10 == 0:
                    torch.save(self.model.state_dict(), self.model_path)
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, step, self.params.steps,
                               loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (epoch+1)))
            torch.save(self.model.state_dict(), self.model_path)

    def test_MDNRNN(self, N):
        filelist, N = self.get_filelist(N)
        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        for epoch in range(self.params.epochs):
            hidden = self.model.init_hidden(10)
            for step in range(self.params.steps):
                z, action, rew, done = self.random_batch(filelist, self.params.batch_size)
                cur_z = z[:, :-1, :]
                #cur_a = action[:, :-1, :]
                #print(cur_a)
                nex_z = z[:, 1:, :]
                #rnn_input = np.concatenate([cur_z, cur_a], axis=2)
                rnn_in = torch.tensor(cur_z, dtype=torch.float32)
                rnn_out = torch.tensor(nex_z, dtype=torch.float32)
                hidden = self.detach(hidden)
                (pi, mu, sigma), hidden = self.model(rnn_in, hidden)
                loss = self.criterion(rnn_out, pi, mu, sigma)
                # self.optimizer.zero_grad()
                # loss.backward()
                train_loss += loss.item()
                # self.optimizer.step()
                if step % 10 == 0:
                    print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, step, self.params.steps,
                        loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (epoch + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--N', default=100, help='number of episodes to use to train')
    args = parser.parse_args()
    N = int(args.N)
    model_path = ('models/' + "mdnrnn" + '.pt')
    params = Params('params/' + "mdnrnn" + '.json')
    MDNRNN = MDNRNNTrainer(params=params, model_path=model_path)
    MDNRNN.train_MDNRNN(N)
    #MDNRNN.test_MDNRNN(N)

