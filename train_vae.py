""" Training VAE """
import os
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import VAE
from params import Params
DIR_NAME = './data/rollout_worldmodel/'
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64

class VAETrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VAE.VAE(input_shape=3, latent_size=self.params.latent_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.dataset, self.N = self.import_data(self.params.N, self.params.M)
        self.dataset_train = self.dataset[:-700]
        self.dataset_test = self.dataset[-700:]
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.params.batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.params.batch_size, shuffle=True, num_workers=2)

    def import_data(self, N, M):
        filelist = os.listdir(DIR_NAME)
        filelist = [x for x in filelist if x != '.DS_Store']
        filelist.sort()
        length_filelist = len(filelist)
        if length_filelist > N:
            filelist = filelist[:N]
        if length_filelist < N:
            N = length_filelist

        data = np.zeros((M * N, 3, SCREEN_SIZE_X, SCREEN_SIZE_Y), dtype=np.float32)
        idx = 0
        file_count = 0

        for file in filelist:
            try:
                new_data = np.load(DIR_NAME + file)['obs']
                new_data = torch.tensor(new_data, dtype=torch.float32)
                new_data = new_data.squeeze(dim=1)
                new_data = np.transpose(new_data, (0, 3, 1, 2))
                data[idx:(idx + M), :, :, :] = new_data

                idx = idx + M
                file_count += 1

                if file_count % 50 == 0:
                    print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
            except Exception as e:
                print(e)
                print('Skipped {}...'.format(file))

        print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
        data = torch.tensor(data)

        return data, N

    def loss_function(self, recon_x, x, mu, logvar):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, size_average=False)
        #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - (logvar).exp())
        return BCE + KLD

    def vae_train(self, epoch):
        """ One training epoch """
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def vae_test(self):
        """ One test epoch """
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        #self.dataset_test.load_next_buffer()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def run(self):
        epochs = self.params.epochs
        for epoch in range(epochs):
            self.vae_train(epoch)
            torch.save(self.model.state_dict(), self.model_path)

if __name__ == '__main__':
    params = Params('params/' + "vae" + '.json')
    model_path = ('models/' + "vae" + '.pt')
    VAETrainer = VAETrainer(params=params, model_path=model_path)
    VAETrainer.run()
    #VAETrainer.vae_test()
