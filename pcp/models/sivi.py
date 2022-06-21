import os
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MIN = -20.
LOG_SIG_MAX = 2.
EPS = 1e-6


class SIVI_NN(nn.Module):
    def __init__(self, x_dim, xi_dim, y_dim, z_dim, H, K, device):
        super(SIVI_NN, self).__init__()
        self.x_dim, self.y_dim, self.z_dim, self.xi_dim = x_dim, y_dim, z_dim, xi_dim
        self.encoder_fc = nn.Sequential(nn.Linear(x_dim + xi_dim, H),
                                        nn.ReLU(),
                                        nn.Linear(H, H),
                                        nn.ReLU())

        self.encoder_mean = nn.Linear(H, z_dim)
        self.encoder_logstd = nn.Linear(H, z_dim)

        self.decoder_fc = nn.Sequential(nn.Linear(x_dim + z_dim, H),
                                     nn.ReLU(),
                                     nn.Linear(H, H),
                                     nn.ReLU())

        self.decoder_mean = nn.Linear(H, y_dim)
        self.decoder_logstd = nn.Linear(H, y_dim)

        self.K = K
        self.device = device
        self.max_log_prob = 10.

    def forward(self, x):
        z, _ = self._forward(x)

        y_h = self.decoder_fc(torch.cat((x, z), dim=-1))
        y_mean = self.decoder_mean(y_h)
        y_std = self.decoder_logstd(y_h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()
        y = Normal(y_mean, y_std).sample()
        return y

    def loss(self, x, y, beta=1.0):
        z, prob_main = self._forward(x)
        prob_aux = self._entropy(x, z)
        log_q_z = torch.log((prob_main + prob_aux + EPS) / (self.K + 1))

        prior_normal = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_p_z = prior_normal.log_prob(z)

        y_h = self.decoder_fc(torch.cat((x, z), dim=-1))
        y_mean = self.decoder_mean(y_h)
        y_std = self.decoder_logstd(y_h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()
        y_normal = Normal(y_mean, y_std)

        recon_loss = y_normal.log_prob(y)

        loss = - (recon_loss.mean() + (log_p_z - log_q_z).mean() * beta)
        return loss

    def _forward(self, x):
        xi = torch.randn((1, self.xi_dim), device=self.device).repeat(x.shape[0], 1)
        h = torch.cat((x, xi), -1)
        h = self.encoder_fc(h)
        z_mean = self.encoder_mean(h)
        z_std = self.encoder_logstd(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        z_normal = Normal(z_mean, z_std)
        z_sample = z_normal.rsample()
        log_prob = z_normal.log_prob(z_sample).clamp(-self.max_log_prob, self.max_log_prob)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return z_sample, log_prob.exp()

    def _entropy(self, x, z):
        M, _ = x.shape
        x = torch.repeat_interleave(x, self.K, dim=0)
        xi = torch.randn((self.K, self.xi_dim), device=self.device).repeat(M, 1)
        # xi = torch.normal(torch.zeros([M * rep, self.noise_dim]),
        #                  torch.ones([M * rep, self.noise_dim]), device=self.device)

        hidden = self.encoder_fc(torch.cat((x, xi), axis=-1))
        z_te_mean = self.encoder_mean(hidden)
        z_te_std = self.encoder_logstd(hidden).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()
        z_te_normal = Normal(z_te_mean, z_te_std)

        z = torch.repeat_interleave(z, self.K, dim=0)

        log_prob = z_te_normal.log_prob(z).clamp(-self.max_log_prob, self.max_log_prob)
        log_prob = log_prob.sum(dim=-1, keepdim=True).view(M, self.K)
        prob = log_prob.exp().sum(dim=-1, keepdim=True)

        return prob


# Vanilla GAN model for y|x
class SIVI(object):
    def __init__(self, x_dim, y_dim, z_dim, H, batch_size, device,
                 lr=1e-3):
        self.z_dim = z_dim
        self.input_dim = x_dim + z_dim
        self.batch_size = batch_size
        self.model = SIVI_NN(x_dim=x_dim,
                             xi_dim=5,
                             y_dim=y_dim,
                             z_dim=z_dim,
                             H=H, K=25, device=device).to(device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def _train_step(self, dataset, batch_size):
        x, y = dataset.sample(batch_size)

        loss = self.model.loss(x, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def train(self, dataset, batch_size, n_epochs):
        iter_per_epoch = dataset.n_samples // batch_size

        loss_stats = []
        for epoch in range(n_epochs + 1):

            loss_epoch = []
            for _ in range(iter_per_epoch):
                loss = self._train_step(dataset, batch_size)
                loss_epoch.append(loss)

            loss_stats.append(np.mean(loss_epoch))

            if epoch % 10 == 0:
                print(f"SIVI_Model: Epoch {epoch} Loss {loss_stats[-1]}")

    def sample(self, x):
        return self.model(x)


