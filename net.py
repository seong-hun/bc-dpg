import torch
import torch.nn as nn

from sklearn.preprocessing import PolynomialFeatures


class QNet(nn.Module):
    def __init__(self, x_size, u_size, lr=1e-4, device="cpu"):
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

    def forward(self, x, u):
        xu = torch.cat((x, u), 1)
        return self.model(xu)


class PolyNet(nn.Module):
    def __init__(self, x_size, u_size, lr,
                 degree=1, include_bias=False, device="cpu"):
        super().__init__()
        self.model = nn.Linear(x_size, u_size)
        self.feature = PolynomialFeatures(
            degree=degree, include_bias=include_bias)

        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        x = torch.tensor(self.feature.fit_transform(x)).float()
        return self.model(x)
