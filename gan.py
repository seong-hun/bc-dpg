from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import fym.logging as logging

device = torch.device('cpu')


class LossWrapper:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, x, y):
        target_tensor = self.get_target_tensor(x, y)
        return self.loss(x, target_tensor)

    def get_target_tensor(self, x, y):
        if y:
            target_tensor = torch.tensor(1.0)
        else:
            target_tensor = torch.tensor(0.0)
        return target_tensor.expand_as(x)


class GAN():
    def __init__(self, lr, x_size, u_size, z_size, lambda_l1=0.01):
        self.z_size = z_size
        self.lambda_l1 = lambda_l1
        self.net_d = Discriminator(x_size=x_size, u_size=u_size)
        self.net_g = Generator(x_size=x_size, u_size=u_size, z_size=z_size)

        self.criterion = LossWrapper(nn.MSELoss())
        self.criterion_l1 = nn.L1Loss()

        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=lr)
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr)

    def set_input(self, data):
        self.real_x, self.real_u = (data[i].to(device) for i in (0, 1))

    def forward(self, x):
        zx = torch.cat((torch.randn(len(x), self.z_size), x), 1)
        self.fake_u = self.net_g(zx)  # G(x)

    def get_action(self, x):
        x = torch.tensor(x).float()
        zx = torch.cat((torch.randn(len(x), self.z_size), x), 1)
        return self.net_g(zx).detach().numpy()  # G(x)

    def train(self):
        self.forward(self.real_x)  # Compute fake control input

        # Train Discriminator
        self.optimizer_d.zero_grad()

        # Fake
        fake_xu = torch.cat((self.real_x, self.fake_u), 1)
        pred_fake = self.net_d(fake_xu.detach())
        self.loss_d_fake = self.criterion(pred_fake, False)

        # Real
        real_xu = torch.cat((self.real_x, self.real_u), 1)
        pred_real = self.net_d(real_xu.detach())
        self.loss_d_real = self.criterion(pred_real, True)

        self.loss_d = (self.loss_d_fake + self.loss_d_real) * 0.5
        self.loss_d.backward()

        self.optimizer_d.step()

        # Train Generator
        self.optimizer_g.zero_grad()
        pred_fake = self.net_d(fake_xu)
        self.loss_g = self.criterion(pred_fake, True)
        # self.loss_g += (
        #     self.criterion_l1(self.fake_u, self.real_u) * self.lambda_l1)
        self.loss_g.backward()

        self.optimizer_g.step()

    def share_memory(self):
        self.net_d.share_memory()
        self.net_g.share_memory()

    def save(self, epoch, savepath):
        torch.save({
            "epoch": epoch,
            "net_d": self.net_d.state_dict(),
            "net_g": self.net_g.state_dict(),
        }, savepath)

    def load(self, loadpath):
        data = torch.load(loadpath)
        self.net_d.load_state_dict(data["net_d"])
        self.net_g.load_state_dict(data["net_g"])


class Discriminator(nn.Module):
    def __init__(self, x_size, u_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(u_size + x_size, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, xu):
        out = self.model(xu)
        return out


class Generator(nn.Module):
    def __init__(self, x_size, u_size, z_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_size + x_size, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, u_size),
        )
        # self.model = nn.Sequential(
        #     nn.Linear(z_size + x_size, 128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, u_size),
        # )

    def forward(self, zx):
        out = self.model(zx)
        return out


class DictDataset(Dataset):
    def __init__(self, file_names, keys):
        if isinstance(file_names, str):
            file_names = (file_names, )

        data_all = [logging.load(name) for name in file_names]
        self.keys = keys if not isinstance(keys, str) else (keys, )
        self.data = {
            k: torch.cat([
                torch.tensor(data[k]).float()
                for data in data_all])
            for k in self.keys
        }
        self.len = len(self.data[self.keys[0]])

    def __getitem__(self, idx):
        return [self.data[k][idx] for k in self.keys]

    def __len__(self):
        return self.len


def get_dataloader(sample_files, keys=("state", "action"), **kwargs):
    return DataLoader(DictDataset(sample_files, keys), **kwargs)
