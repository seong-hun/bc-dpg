import numpy as np
import numpy.linalg as nla
from collections import deque
import random

import torch
import torch.nn as nn

import fym.logging as logging

import utils
import gan


class BaseAgent:
    def __init__(self, env, theta_init):
        self.clock = env.clock
        self.saturation = env.system.saturation
        self.trim_x = env.trim_x
        self.trim_u = env.trim_u
        self.m = self.trim_u.size
        self.n_phi = self.phi(self.trim_x).size
        self.theta = theta_init * np.random.randn(self.n_phi, self.m)
        self.noise = False

    def add_noise(self, scale, tau):
        self.noise = {"scale": scale, "tau": tau}

    def get_action(self, obs):
        if not self.noise:
            return self.get_behavior(self.theta, obs)
        else:
            time = self.clock.get()
            theta = self.theta + (
                self.noise["scale"]
                * np.exp(-time/self.noise["tau"])
                * np.random.randn()
            )
            return self.get_behavior(theta, obs)

    def phi(self, x, deg=[1]):
        return utils.get_poly(x, deg=deg)

    def get_behavior(self, theta, x):
        """If ``x = self.trim_x``, then ``del_u = 0``. This is ensured by
        the structure of ``phi`` which has no constant term. Also,
        the behavior policy should always be saturated by the control limits
        defined by the system."""
        del_ub = theta.T.dot(self.phi(x))
        return self.saturation(self.trim_u + del_ub) - self.trim_u


class COPDAC(BaseAgent):
    "Compatible off-policy deterministic actor-critc"
    def __init__(self, env, lrw, lrv, lrtheta, w_init, v_init, theta_init,
                 maxlen, batch_size):
        self.saturation = env.system.saturation
        self.trim_x = env.trim_x
        self.trim_u = env.trim_u
        self.m = self.trim_u.size
        self.clock = env.clock
        self.lrw = lrw
        self.lrv = lrv
        self.lrtheta = lrtheta
        self.batch_size = batch_size

        self.n_phi = self.phi(self.trim_x).size
        self.theta = theta_init * np.random.randn(self.n_phi, self.m)
        self.w = w_init * np.random.randn(
            self.phi_w(self.trim_x, self.trim_u, self.theta).size)
        self.v = v_init * np.random.randn(self.phi_v(self.trim_x).size)

        self.is_gan = False
        self.is_reg = False

    def get_action(self, obs, noise_scale=0):
        time = self.clock.get()
        theta = self.theta + noise_scale * np.exp(-time/2) * np.random.randn()
        return self.get_behavior(theta, obs)

    def phi_v(self, x, deg=2):
        # return utils.get_poly(x, deg=deg)
        return np.hstack((x, x**2))

    def phi_w(self, x, u, theta):
        return self.dpi_dtheta(x).dot(u - np.dot(self.phi(x), theta))

    def phi_q(self, x, u, theta):
        return np.hstack((self.phi_c(x, u, theta), self.phi_v(x)))

    def dpi_dtheta(self, x):
        return np.kron(np.eye(self.m), self.phi(x)).T

    def load(self, path):
        data = logging.load(path)
        self.load_weights(data)
        return data["epoch"], data["i"]

    def load_weights(self, data):
        self.theta = data["theta"][-1]
        self.w = data["w"][-1]
        self.v = data["v"][-1]

    def set_gan(self, path, lrg):
        self.gan = gan.GAN(x_size=4, u_size=4, z_size=100)
        self.gan.load(path)
        self.gan.eval()
        self.lrg = lrg
        self.is_gan = True

    def set_reg(self, lrc):
        self.lrc = lrc
        self.is_reg = True

    def set_input(self, data):
        self.data = [np.array(d, dtype=np.float) for d in data]

    def backward(self):
        grad_w = np.zeros_like(self.w)
        grad_v = np.zeros_like(self.v)
        grad_theta = np.zeros_like(self.theta.ravel())
        delta = 0
        for x, bu, reward, nx in zip(*self.data):
            us = self.get_behavior(self.theta, nx)
            tderror = (
                1 * (
                    self.w.dot(self.phi_w(nx, us, self.theta))
                    + self.v.dot(self.phi_v(nx))
                )
                + reward
                - (
                    self.w.dot(self.phi_w(x, bu, self.theta))
                    + self.v.dot(self.phi_v(x))
                )
            )

            grad_w += - tderror * self.phi_w(x, bu, self.theta)
            grad_v += - tderror * self.phi_v(x)
            dpdt = self.dpi_dtheta(x)
            grad_theta += dpdt.dot(dpdt.T).dot(self.w)

            delta += tderror

        n = len(self.data[0])
        self.grad_w = - grad_w / n
        self.grad_v = - grad_v / n
        self.grad_theta = - grad_theta.reshape(self.theta.shape) / n
        self.delta = delta / n

    def step(self):
        if np.abs(self.delta) > 0.005:
            self.w = self.w + self.lrw * self.grad_w
            self.v = self.v + self.lrv * self.grad_v
            add_grad = 0
            if self.is_gan:
                add_grad += - self.lrg * self.gan_grad()
            if self.is_reg:
                add_grad += - self.lrc * self.theta

            # print(np.abs(add_grad).max(), np.abs(self.lrc*self.theta).max())
            self.theta = (
                self.theta + self.lrtheta * self.grad_theta
                + add_grad)

    def gan_grad(self):
        # x, u, _, _ = [torch.tensor(d).float() for d in self.data]
        x = self.data[0]
        u = np.vstack([self.get_behavior(self.theta, xi) for xi in x])
        x = torch.tensor(x).float()
        xu = torch.tensor(np.hstack((x, u))).float()
        xu.requires_grad = True
        loss_d = self.gan.net_d(xu).mean()
        loss_d.backward()
        grad_du = np.array(xu.grad[:, 4:], dtype=np.float)  # dD / du
        grad = 0
        for xi, grad_dui in zip(x, grad_du):
            dpdt = self.dpi_dtheta(xi)  # du / dtheta
            grad += dpdt.dot(grad_dui)
        return grad.reshape(self.theta.shape) / len(x)

    def train(self):
        self.backward()
        self.step()


class RegCOPDAC(COPDAC):
    "Regulated compatible off-policy deterministic actor-critc"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.backward()
        self.theta = self.theta - self.lrc * self.theta
        self.step()
