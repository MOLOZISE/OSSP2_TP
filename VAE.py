import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, input_shape=3, latent_size=32):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = input_shape
        self.conv1 = nn.Conv2d(input_shape, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logvar = nn.Linear(2*2*256, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, input_shape=3, latent_size=32):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = input_shape

        self.fc1 = nn.Linear(latent_size, 1*1*1024)
        self.deconv1 = nn.ConvTranspose2d(1*1*1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, input_shape, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, input_shape=3, latent_size=32):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(input_shape, latent_size)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_z = self.decoder(z)
        return recon_z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        recon_z = self.decoder(z)
        return recon_z
