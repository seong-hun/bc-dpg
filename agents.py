import numpy as np
import numpy.linalg as nla
from collections import deque
import random

from utils import get_poly


class Agent:
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
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

        self.n_phi = self.phi(self.trim_x).size
        self.theta = theta_init * np.random.randn(self.n_phi, self.m)
        self.w = w_init * np.random.randn(
            self.phi_w(self.trim_x, self.trim_u, self.theta).size)
        self.v = v_init * np.random.randn(self.phi_v(self.trim_x).size)

    def get_action(self, obs, noise_scale=0):
        time = self.clock.get()
        theta = self.theta + noise_scale * np.exp(-time/2) * np.random.randn()
        return self.get_behavior(theta, obs)

    def append(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        grad_w = np.zeros_like(self.w)
        grad_v = np.zeros_like(self.v)
        grad_theta = np.zeros_like(self.theta.ravel())
        for b in batch:
            x, bu, reward, nx = b
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
            if np.abs(tderror) > 0.0001:
                grad_w += - tderror * self.phi_w(x, bu, self.theta)
                grad_v += - tderror * self.phi_v(x)
                dpdt = self.dpi_dtheta(x)
                grad_theta += dpdt.dot(dpdt.T).dot(self.w)

        self.w = self.w - self.lrw * grad_w / len(batch)
        self.v = self.v - self.lrv * grad_v / len(batch)
        self.theta = (
            (1 - self.lrtheta) * self.theta
            - self.lrtheta * grad_theta.reshape(self.theta.shape) / len(batch)
        )

    def phi(self, x, deg=[1]):
        return get_poly(x, deg=deg)

    def phi_v(self, x, deg=2):
        return get_poly(x, deg=deg)
        # return np.hstack((x, x**2))

    def phi_w(self, x, u, theta):
        return self.dpi_dtheta(x).dot(u - np.dot(self.phi(x), theta))

    def phi_q(self, x, u, theta):
        return np.hstack((self.phi_c(x, u, theta), self.phi_v(x)))

    def dpi_dtheta(self, x):
        return np.kron(np.eye(self.m), self.phi(x)).T

    def get_behavior(self, theta, x):
        """If ``x = self.trim_x``, then ``del_u = 0``. This is ensured by
        the structure of ``phi`` which has no constant term. Also,
        the behavior policy should always be saturated by the control limits
        defined by the system."""
        del_ub = theta.T.dot(self.phi(x))
        return self.saturation(self.trim_u + del_ub) - self.trim_u

    def load_weights(self, datapath):
        import fym.logging as logging

        data = logging.load(datapath)
        self.theta = data["theta"]
        self.w = data["w"]
        self.v = data["v"]
