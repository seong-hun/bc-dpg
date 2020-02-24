import numpy as np
import numpy.linalg as nla
from collections import deque
import random

import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures

import fym.logging as logging

import utils
import gan
import net


class BaseInnerCtrl:
    def __init__(self, xdim, udim):
        self.xtrim = np.zeros(xdim)
        self.utrim = np.zeros(udim)

    def set_trim(self, xtrim, utrim):
        self.xtrim = xtrim
        self.utrim = utrim

    def get(self, t, x):
        raise NotImplementedError


class Linear(BaseInnerCtrl):
    def __init__(self, xdim, udim):
        super().__init__(xdim, udim)
        self.theta = np.zeros((xdim, udim))
        self.noise = None

    def set_param(self, theta):
        if np.ndim(theta) == 0:
            theta = theta * np.ones_like(self.theta)

        assert np.ndim(theta) == np.ndim(self.theta)

        self.theta = theta

    def get(self, t, x):
        theta = self.theta + self.get_noise(t)
        return (x - self.xtrim).dot(theta) + self.utrim

    def get_noise(self, t):
        if self.noise is not None:
            return self.noise.get(t)
        else:
            return 0

    def add_noise(self, noise):
        self.noise = noise
        self.noise.shape = self.theta.shape


class BaseAgent:
    """Generate trajectories of behavior policy"""
    def __init__(self, env, theta_init):
        self.name = "BaseAgent"

        self.clock = env.clock
        self.limits = np.vstack([
            env.system.control_limits[k]
            for k in ("delt", "dele", "eta1", "eta2")
        ]).T
        self.get_behavior = env.get_behavior
        self.phi = env.phi

        self.trim_x = env.trim_x
        self.trim_u = env.trim_u
        self.m = self.trim_u.size

        self.n_phi = self.phi(self.trim_x).shape[1]
        self.theta = theta_init * np.random.randn(self.n_phi, self.m)
        self.noise = False

    def add_noise(self, scale, tau):
        self.noise = {"scale": scale, "tau": tau}

    def get_action(self, obs):
        if not self.noise:
            return self.theta
        else:
            time = self.clock.get()
            theta = self.theta + (
                self.noise["scale"]
                * np.exp(-time/self.noise["tau"])
                * np.random.randn()
            )
            return theta


class COPDAC(nn.Module):
    "Compatible off-policy deterministic actor-critc"
    def __init__(self, env, net_option, **kwargs):
        super().__init__()
        self.name = "COPDAC"

        self.net_q = self.set_net(net_option, "net_q")
        self.net_pi = self.set_net(net_option, "net_pi")

        self.addons = []

    def set_net(self, option, name):
        opt = option[name]
        return getattr(net, opt["class"])(**opt["option"])

    def set_input(self, data):
        self.data = data
        # self.data = [np.array(d, dtype=np.float) for d in data]

    def train(self):
        x, bu, reward, nx = self.data

        self.net_q.zero_grad()
        us = self.net_pi(nx)
        with torch.no_grad():
            target = self.net_q(nx, us) + reward
        loss = self.net_q.criterion(self.net_q(x, bu), target)
        loss.backward()
        self.net_q.optimizer.step()
        self.net_q.loss = loss.detach().numpy()

        self.net_pi.zero_grad()
        us = self.net_pi(x)
        loss = self.net_q(x, us).mean()
        loss.backward()
        self.net_pi.optimizer.step()
        self.net_pi.loss = loss.detach().numpy()

    def get_msg(self):
        return "\t".join([
            f"{name} loss: {loss:5.4f}"
            for name, loss in self.get_losses().items()
        ])

    def get_name(self):
        return "-".join((self.name, *sorted(self.addons)))

    def load(self, path):
        data = logging.load(path)
        self.load_weights(data)
        mode = "r+"
        return int(data["epoch"][-1] + 1), int(data["i"][-1] + 1), mode

    def get_losses(self):
        loss = {name: module.loss for name, module in self.named_children()}
        if "gan" in self.addons:
            if self.gan_type == "d":
                loss.update(gan=self.gan_loss.detach().numpy())
            elif self.gan_type == "g":
                loss.update(gan=self.gan_loss)
        return loss

    def set(self):
        # Learning rates
        self.lrw = lrw
        self.lrv = lrv
        self.lrtheta = lrtheta

        self.batch_size = batch_size

        # Initialize COPDAC parameters
        self.poly_v = PolynomialFeatures(degree=v_deg, include_bias=False)
        self.w = w_init * np.random.randn(
            self.phi_w(
                np.atleast_2d(self.trim_x),
                np.atleast_2d(self.trim_u),
                self.theta
            ).size)
        self.v = v_init * np.random.randn(self.phi_v(self.trim_x).size)
        self.theta = theta_init * np.random.randn(self.n_phi, self.m)

        # Add-ons
        self.addons = []

        self.QNet = net.QNet(self.trim_x.size, self.trim_u.size)

    def phi_v(self, x):
        poly = self.poly_v.fit_transform(np.atleast_2d(x))
        if np.ndim(x) == 1:
            return poly[0]
        else:
            return poly

    def phi_w(self, x, u, theta):
        # (batch, 1, m)
        y = (u - self.get_behavior(theta, x))[:, None, :]  # (batch, 1, m)
        # (batch, 1, m) * (batch, m * n_phi, m) -> (batch, m * n_phi)
        return np.sum(y * self.dpi_dtheta(x), axis=-1)

    def dpi_dtheta(self, x):
        eye = np.eye(self.m)[None, :, :]  # (1, m, m)
        phi = self.phi(x)[:, :, None]  # (batch, n_phi, 1)
        return np.kron(eye, phi)  # (batch, m * n_phi, m)

    def get_Q(self, x, u, w, v, theta):
        Q = self.phi_w(x, u, theta).dot(w) + self.phi_v(x).dot(v)
        return Q.reshape(-1, 1)

    def step(self):
        if np.abs(self.delta) > 0.001:
            self.w = self.w - self.lrw * self.grad_w
            self.v = self.v - self.lrv * self.grad_v

            add_grad = 0

            if "gan" in self.addons:
                add_grad += - self.lrg * self.gan_grad()
                # print(np.abs(self.theta).max(), np.abs(add_grad).max())
            if "reg" in self.addons:
                add_grad += - self.lrc * self.theta

            new_theta = self.theta - self.lrtheta * self.grad_theta + add_grad

            if "const" in self.addons:
                if not self.in_bound(new_theta):
                    new_theta = self.theta

            self.theta = new_theta

            if "dual_Q" in self.addons:
                self.QNet.optimizer.step()

    def load_weights(self, data):
        self.theta = data["theta"][-1]
        self.w = data["w"][-1]
        self.v = data["v"][-1]

    def in_bound(self, theta):
        x = self.data[0]
        u = self.phi(x).dot(theta) + self.trim_u

        in_rate = np.logical_and(
            self.limits[0] <= u,
            u <= self.limits[1]
        ).all(1).mean()

        if in_rate == 1:
            return True
        else:
            return False

    def set_gan(self, path, lrg, gan_type="g"):
        self.addons.append("gan" + "_" + gan_type)

        self.gan = gan.GAN(x_size=4, u_size=4, z_size=100)
        self.gan.load(path)
        self.gan.eval()
        self.lrg = lrg
        self.gan_type = gan_type

    def set_reg(self, lrc):
        self.addons.append("reg")
        self.lrc = lrc

    def set_const(self):
        self.addons.append("const")

    def gan_grad(self):
        if self.gan_type == "d":
            x = self.data[0]
            u = np.vstack([self.get_behavior(self.theta, xi) for xi in x])
            xu = torch.tensor(np.hstack((x, u))).float()
            xu.requires_grad = True
            self.gan_loss = self.gan.criterion(self.gan.net_d(xu).mean(), True)
            self.gan_loss.backward()
            grad_du = np.array(xu.grad[:, 4:], dtype=np.float)  # dD / du
            grad = 0
            for xi, grad_dui in zip(x, grad_du):
                dpdt = self.dpi_dtheta(xi)  # du / dtheta
                grad += - dpdt.dot(grad_dui)

        elif self.gan_type == "g":
            x = self.torch_data[0]
            u_cand = torch.stack([
                self.gan.get_action(x, to="torch") for _ in range(25)
            ])
            Q_cand = torch.cat([self.QNet(x, u) for u in u_cand], 1)
            us = u_cand.gather(
                0, Q_cand.argmin(1).view(1, -1, 1).repeat(1, 1, 4))[0]
            # u = self.get_behavior(self.theta.T, x)
            grad = 0
            loss = 0
            for xi, ui in zip(x, us):
                dpdt = self.dpi_dtheta(xi)
                e = dpdt.T.dot(self.theta.T.ravel()) - ui.numpy()
                loss += np.sum(e ** 2) / 2
                grad += dpdt.dot(e)

            self.gan_loss = loss / len(x)

        grad = grad.reshape(self.theta.shape, order="F") / len(x)
        return grad

    def phi(self, x_trimmed):
        poly = self.poly_phi.fit_transform(np.atleast_2d(x_trimmed))
        return poly

    def get(self, theta, x_trimmed):
        """output: trimmed and saturated u"""
        assert np.ndim(x_trimmed) == 2, "dim(x) = (batch, size_x)"
        u_trimmed = self.phi(x_trimmed).dot(theta)
        # return self.saturation(self.trim_u + u_trimmed) - self.trim_u
        return u_trimmed


class RegCOPDAC(COPDAC):
    "Regulated compatible off-policy deterministic actor-critc"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.backward()
        self.theta = self.theta - self.lrc * self.theta
        self.step()


if __name__ == "__main__":
    import envs
    import main

    PARAMS = main.PARAMS

    env = envs.FixedParamEnv(
        **{**PARAMS["sample"]["FixedParamEnv"], "logging_off": True})
    agent = COPDAC(
        **PARAMS["train"]["COPDAC"]
    )
