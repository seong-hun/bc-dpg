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


class COPDAC(BaseAgent):
    "Compatible off-policy deterministic actor-critc"
    def __init__(self, env, lrw, lrv, lrtheta, w_init, v_init, theta_init,
                 v_deg, maxlen, batch_size):
        super().__init__(env, theta_init)
        self.name = "COPDAC"

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

    def set_input(self, data):
        self.torch_data = data
        self.data = [np.array(d, dtype=np.float) for d in data]

    def train(self):
        self.backward()  # calculate gradients
        self.step()  # update the parameters

    def backward(self):
        x, bu, reward, nx = self.data

        us = self.get_behavior(self.theta, nx)
        tderror = (
            1 * self.get_Q(nx, us, self.w, self.v, self.theta)
            + reward
            - self.get_Q(x, bu, self.w, self.v, self.theta)
        )
        grad_w = - tderror * self.phi_w(x, bu, self.theta)
        grad_v = - tderror * self.phi_v(x)
        dpdt = self.dpi_dtheta(x)  # (b, m * n_phi, m) or (m * n_phi, m)
        grad_theta = np.einsum("bij,bkj->bik", dpdt, dpdt).dot(self.w)

        self.grad_w = np.mean(grad_w, axis=0)
        self.grad_v = np.mean(grad_v, axis=0)
        self.grad_theta = np.mean(grad_theta, axis=0).reshape(
            self.theta.shape, order="F")
        self.delta = np.mean(tderror, axis=0)

        # Train dual Q-network
        if "dual_Q" in self.addons:
            self.QNet.zero_grad()
            x, bu, reward, nx = self.torch_data
            us = torch.tensor(self.get_behavior(self.theta, nx)).float()
            current_Q = self.QNet(x, bu)
            target_Q = self.QNet(nx, us) + reward
            self.dual_q_loss = self.QNet.criterion(target_Q, current_Q)
            self.dual_q_loss.backward()

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

    def load(self, path):
        data = logging.load(path)
        self.load_weights(data)
        return int(data["epoch"][-1] + 1), int(data["i"][-1] + 1)

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

    def get_name(self):
        return "-".join((self.name, *sorted(self.addons)))

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

    def get_losses(self):
        loss = {"delta": self.delta}
        if "gan" in self.addons:
            if self.gan_type == "d":
                loss.update(gan=self.gan_loss.detach().numpy())
            elif self.gan_type == "g":
                loss.update(gan=self.gan_loss)
        return loss


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

    env = envs.BaseEnv(
        initial_perturb=[1, 0.0, 0, np.deg2rad(10)],
        dt=0.01, max_t=40, solver="rk4",
        ode_step_len=1
    )
    agent = COPDAC(
        env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2,
        w_init=0.03, v_init=0.03, theta_init=0,
        maxlen=100, batch_size=16
    )
