from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import fym.logging as logging


class Discriminator(nn.Module):
    def __init__(self, x_size, u_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(u_size + x_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1),
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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, u_size),
        )

    def forward(self, zx):
        out = self.model(zx)
        return out


class GAN():
    def __init__(self, lr, x_size, u_size, z_size,
                 use_cuda=False, lambda_l1=0.01):
        self.z_size = z_size
        self.lambda_l1 = lambda_l1
        self.net_d = Discriminator(x_size=x_size, u_size=u_size)
        self.net_g = Generator(x_size=x_size, u_size=u_size, z_size=z_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.initialize(self.net_d)
        self.initialize(self.net_g)

        self.criterion = LossWrapper(nn.MSELoss()).to(self.device)
        self.criterion_l1 = nn.L1Loss()

        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=lr)
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr)

    def initialize(self, net):
        net.to(self.device)
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.2)
                nn.init.constant_(module.bias, 0)

    def set_input(self, data):
        self.real_x, self.real_u = (data[i].to(self.device) for i in (0, 1))

    def forward(self, x):
        z = torch.randn(len(x), self.z_size).to(self.device)
        zx = torch.cat((z, x), 1)
        self.fake_u = self.net_g(zx)  # G(x)

    def get_action(self, x):
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
        self.loss_g += (
            self.criterion_l1(self.fake_u, self.real_u) * self.lambda_l1)
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
        data = torch.load(loadpath, map_location=self.device)
        self.net_d.load_state_dict(data["net_d"])
        self.net_g.load_state_dict(data["net_g"])
        return data["epoch"]

    def eval(self):
        self.net_d.eval()
        self.net_g.eval()


class LossWrapper(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.loss = loss

    def __call__(self, x, y):
        target_tensor = self.get_target_tensor(x, y)
        return self.loss(x, target_tensor)

    def get_target_tensor(self, x, y):
        if y:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)


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
