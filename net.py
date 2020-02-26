import torch
import torch.nn as nn

from sklearn.preprocessing import PolynomialFeatures


class QNet(nn.Module):
    def __init__(self, x_size, u_size, lr=1e-4, gamma=0.99, device="cpu"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_size + u_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1),
        )

        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma)

    def forward(self, x, u):
        xu = torch.cat((x, u), 1)
        return self.model(xu)


class PolyNet(nn.Module):
    def __init__(self, x_size, u_size, lr,
                 degree=1, include_bias=False, device="cpu"):
        super().__init__()
        self.feature = PolynomialFeatures(
            degree=degree, include_bias=include_bias)
        sample_feature = self.feature.fit_transform(torch.zeros(1, x_size))
        self.model = nn.Linear(sample_feature.shape[1], u_size, bias=False)

        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        x = torch.tensor(self.feature.fit_transform(x)).float()
        return self.model(x)


class BCPi(nn.Module):
    def __init__(self, x_size, u_size, z_size, lr, device="cpu"):
        super().__init__()
        self.z_layer = nn.Linear(1, z_size)
        self.pre_model = nn.Sequential(
            nn.Linear(z_size + x_size, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, u_size),
        )

        self.optimizer = torch.optim.Adam(self.z_layer.parameters(), lr=lr)

    def forward(self, x):
        z = self.z_layer(torch.ones(x.shape[0], 1))
        zx = torch.cat((z, x), 1)
        out = self.pre_model(zx)
        return out


if __name__ == "__main__":
    polynet = PolyNet(4, 4, 1e-3, degree=2)
