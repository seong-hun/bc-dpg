import numpy as np
import numpy.linalg as nla

import fym
import fym.core
import fym.agents.LQR
import fym.logging as logging
from fym.models.aircraft import MorphingLon
from fym.utils.linearization import jacob_analytic

from utils import get_poly


class BaseEnv(fym.core.BaseEnv):
    Q = np.diag([1, 100, 10, 100])
    R = np.diag([50, 1, 1, 1])

    def __init__(self, initial_perturb, **kwargs):
        self.system = MorphingLon([0, 0, 0, 0])
        self.IR_system = fym.core.BaseSystem(0, name="integral reward")
        super().__init__(
            systems_dict={
                "main": self.system,
                "IR": self.IR_system,
            },
            **kwargs
        )

        self.initial_perturb = initial_perturb
        trim_x, trim_u = self.system.get_trim()
        self.trim_x = trim_x
        self.trim_u = trim_u
        self.system.initial_state = trim_x + initial_perturb

    def reset(self, initial_perturb=None):
        if initial_perturb == "random":
            self.system.initial_state = (
                self.trim_x
                + self.initial_perturb
                + [1, 0.05, 0.05, 0.05] * np.random.randn(4)
            )

        super().reset()
        return self.observation()

    def observation(self):
        return self.system.state - self.trim_x

    def step(self, action):
        done = self.clock.time_over()
        time = self.clock.get()
        x = self.observation()
        IR = self.IR_system.state
        u = action + self.trim_u

        if np.any(np.abs(x[(1, 3), ]) > np.deg2rad(30)):
            done = True

        self.update(u)

        next_x = self.observation()
        nIR = self.IR_system.state
        reward = nIR - IR

        info = {
            "time": time,
            "state": x,
            "action": action,
            "reward": reward,
            "next_state": next_x
        }

        return next_x, reward, done, info

    def set_dot(self, time, u):
        x, _ = self.observe_list()

        self.system.dot = self.system.deriv(x, u)
        self.systems_dict["IR"].dot = self.reward(x, u)

    def reward(self, x, u):
        tx = x - self.trim_x
        tu = u - self.trim_u
        return tx.dot(self.Q).dot(tx) + tu.dot(self.R).dot(tu)


class AdpEnv(BaseEnv):
    def __init__(self, initial_perturb, W_init, eta, **kwargs):
        super().__init__(initial_perturb, W_init, **kwargs)
        self.eta = eta
        self.grad_u_phi_c = jacob_analytic(self.phi_c, i=1)

    def reset(self):
        super().reset()
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
        super().set_dot(time, action)
        x, _ = self.observe_list()
        Wb, Wc = action
        bu = self.get_behavior(Wb, x)
        # us = self.get_behavior(W, x)
        grad_Q = np.outer(
            self.phi(x),
            self.grad_u_phi_c(x, bu).T.dot(Wc)
        )

        self.systems_dict["W"].dot = - self.eta * grad_Q
