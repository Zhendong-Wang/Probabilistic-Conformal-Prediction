import os
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Vanilla GAN model for y|x
class GAN(object):
    def __init__(self, x_dim, y_dim, z_dim, H, batch_size, device,
                 d_lr=1e-3, g_lr=3e-4,
                 adam_b1=0.5,
                 adam_b2=0.999,
                 n_disc_batch=2):
        self.z_dim = z_dim
        self.input_dim = x_dim + z_dim
        self.batch_size = batch_size
        self.generator = nn.Sequential(
            nn.Linear(self.input_dim, H),
            # nn.BatchNorm1d(H),
            nn.LeakyReLU(0.1),
            nn.Linear(H, H),
            # nn.BatchNorm1d(H // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(H, y_dim)
        ).to(device)

        self.discriminator = nn.Sequential(
            nn.Linear(x_dim + y_dim, H),
            # nn.BatchNorm1d(H),
            nn.LeakyReLU(0.1),
            nn.Linear(H, H),
            # nn.BatchNorm1d(H // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(H, 1)
        ).to(device)
        
        self.scale_y = False
        self.n_disc_batch = n_disc_batch
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(adam_b1, adam_b2))
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(adam_b1, adam_b2))
        self.ad_loss = nn.BCEWithLogitsLoss()
        self.device = device

    def train_step(self, dataset, batch_size):

        for _ in range(self.n_disc_batch):
            x, y = dataset.sample(batch_size)
            v = torch.cat([x, y], 1)
            z = torch.randn(batch_size, self.z_dim).to(self.device)
            g_input = torch.cat([x, z], 1)
            gen_y = self.generator(g_input)
            w = torch.cat([x, gen_y], 1)

            disc_loss = self.ad_loss(self.discriminator(v), torch.ones(self.batch_size, 1).to(self.device)) + \
                        self.ad_loss(self.discriminator(w.detach()), torch.zeros(self.batch_size, 1).to(self.device))

            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()

        x, _ = dataset.sample(batch_size)
        gen_z = torch.randn(batch_size, self.z_dim).to(self.device)
        gen_y = self.generator(torch.cat([x, gen_z], 1))
        w = torch.cat([x, gen_y], 1)
        gen_loss = self.ad_loss(self.discriminator(w), torch.ones(self.batch_size, 1).to(self.device))

        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        return disc_loss.item(), gen_loss.item()

    def train(self, dataset, batch_size, n_epochs):
        if dataset.scale_y:
            self.scale_y = True
            self.y_scaler = dataset.y_scaler

        iter_per_epoch = dataset.n_samples // batch_size
        disc_stats = []
        gen_stats = []
        for epoch in range(n_epochs+1):

            ### Train ###
            disc_loss_epoch = []
            gen_loss_epoch = []
            for _ in range(iter_per_epoch):
                disc_loss, gen_loss = self.train_step(dataset, batch_size)
                disc_loss_epoch.append(disc_loss)
                gen_loss_epoch.append(gen_loss)

            disc_stats.append(np.mean(disc_loss_epoch))
            gen_stats.append(np.mean(gen_loss_epoch))

            ### Evaluation ###
            if epoch % 10 == 0:
                print(f"GAN: Epoch {epoch} disc_loss {disc_stats[-1]} gen_loss {gen_stats[-1]}")

    def decode(self, x, z=None):
        if z is None:
            z = torch.randn((x.shape[0], self.z_dim)).to(self.device)

        y = self.generator(torch.cat([x, z], 1))
        if self.scale_y:
            y = torch.from_numpy(self.y_scaler.inverse_transform(y.detach().cpu().numpy()))
        return y

    def eval_uncertainty(self, x, num_bins=50):
        ### Not Finished ###
        n = len(x)
        x = torch.repeat_interleave(x, num_bins, dim=0)
        gen_z = torch.randn(n * num_bins, self.z_dim)
        acts = self.generator(torch.cat([x, gen_z], 1)).view(n, num_bins, -1)
        return acts.detach().numpy()

    def sample(self, x):
        return self.decode(x)

    def eval(self):
        self.generator = self.generator.eval()
        self.discriminator = self.discriminator.eval()
