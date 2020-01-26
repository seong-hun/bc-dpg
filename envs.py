import numpy as np
import numpy.linalg as nla

import fym
from fym.core import BaseEnv, BaseSystem
import fym.agents.LQR
import fym.logging as logging
from fym.models.aircraft import MorphingLon
from fym.utils.linearization import jacob_analytic

from utils import get_poly

np.random.seed(1)


class Env(BaseEnv):
    Q = 10 * np.diag([10, 100, 10, 100])
    R = np.diag([1, 1, 0.1, 0.1])

    def __init__(self, initial_state, W_init, eta, **kwargs):
        self.eta = eta
        self.system = MorphingLon(initial_state)
        self.IR_system = BaseSystem(0, name="integral reward")
        super().__init__(
            systems_dict={
                "main": self.system,
                "IR": self.IR_system,
            },
            **kwargs
        )

        trim_x, trim_u = self.system.get_trim()
        self.trim_x = trim_x
        self.trim_u = trim_u

        x = self.system.state
        phi = self.phi(x)
        W_init = W_init * np.random.randn(phi.size, trim_u.size)
        self.W_system = BaseSystem(W_init, name="actor weights")
        self.append_systems({"W": self.W_system})

        self.grad_u_phi_c = jacob_analytic(self.phi_c, i=1)

    def reset(self):
        super().reset()
        self.system.state = self.trim_x + [0, 0, 0, 0.02]
        return self.observation()

    def observation(self):
        return self.observe_list()

    def step(self, action):
        done = self.clock.time_over()
        time = self.clock.get()
        x, dIR, W = self.observe_list()

        Wb, Wc = action
        bu = self.get_behavior(Wb, x)

        if np.any(np.abs(x[(1, 3), ]) > np.deg2rad(30)):
            done = True

        if np.abs(W).max() > 100 or np.abs(Wc).max() > 100:
            done = True

        self.update(action)

        IR = self.IR_system.state
        reward = IR - dIR

        info = {
            "time": time,
            "trim_x": self.trim_x,
            "trim_u": self.trim_u,
            "state": x,
            "control": bu,
            "W": W.ravel(),
            "Wb": Wb.ravel(),
            "Wc": Wc.ravel(),
            "reward": reward,
        }

        return self.observation(), reward, done, info

    def get_behavior(self, Wb, x):
        """If ``x = self.trim_x``, then ``del_u = 0``. This is ensured by
        the structure of ``phi`` which has no constant term. Also,
        the behavior policy should always be saturated by the control limits
        defined by the system."""
        del_ub = Wb.T.dot(self.phi(x))
        return self.system.saturation(self.trim_u + del_ub)

    def get_target(self, Wc, W, x):
        del_u = W.T.dot(self.phi(x))
        return self.system.saturation(self.trim_u + del_u)

    def set_dot(self, time, action):
        x, RI, W = self.observe_list()
        Wb, Wc = action
        bu = self.get_behavior(Wb, x)
        # us = self.get_behavior(W, x)
        grad_Q = np.outer(
            self.phi(x),
            self.grad_u_phi_c(x, bu).T.dot(Wc)
        )

        self.system.dot = self.system.deriv(x, bu)
        self.systems_dict["IR"].dot = self.reward(x, bu)
        self.systems_dict["W"].dot = - self.eta * grad_Q

    def phi_c(self, x, u, deg=2):
        # n = len(get_poly(u, deg=2))
        n = 10
        return get_poly(
            np.hstack((x - self.trim_x, u - self.trim_u)), deg=deg)[:-n]

    def phi(self, x, deg=[1, 2, 3]):
        return get_poly(x - self.trim_x, deg=deg)

    def reward(self, x, u):
        tx = x - self.trim_x
        tu = u - self.trim_u
        return tx.dot(self.Q).dot(tx) + tu.dot(self.R).dot(tu)
