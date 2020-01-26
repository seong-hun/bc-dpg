import numpy as np
import numpy.linalg as nla
from collections import deque
import random

from utils import get_poly


class Agent:
    def __init__(self, env, lr, Wc_init, maxlen, batch_size):
        self.env = env
        self.lr = lr
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

        x_shape = env.trim_x.shape
        u_shape = env.trim_u.shape
        self.Wc = Wc_init * 2 * np.random.randn(
            env.phi_c(np.zeros(x_shape), np.zeros(u_shape)).size)

    def get_action(self, obs):
        _, _, W = obs
        time = self.env.clock.get()
        Wb = W + 0.00 * np.exp(-time/2) * np.random.randn()
        return Wb, self.Wc

    def append(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None

        env = self.env
        batch = random.sample(self.buffer, self.batch_size)
        grad = 0
        for b in batch:
            (x, _, W), action, reward, (nx, _, _) = b
            Wb, _ = action
            bu = env.get_behavior(Wb, x)
            us = env.get_behavior(W, nx)
            del_phi_c = env.phi_c(nx, us) - env.phi_c(x, bu)
            e = self.Wc.dot(del_phi_c) + reward
            grad += e * del_phi_c

        self.Wc = self.Wc - self.lr * grad / nla.norm(grad)
