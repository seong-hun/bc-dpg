import numpy as np
import numpy.linalg as nla

from sklearn.preprocessing import PolynomialFeatures

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

    def __init__(self, initial_perturb=[0, 0, 0, 0], phi_deg=1, **kwargs):
        self.system = MorphingLon(initial_perturb)
        self.IR_system = fym.core.BaseSystem(0, name="integral reward")
        super().__init__(
            systems_dict={
                "main": self.system,
                "IR": self.IR_system,
            },
            name="MorphingLon",
            **kwargs
        )

        self.initial_perturb = initial_perturb
        self.trim_x, self.trim_u = self.system.get_trim()
        self.system.initial_state = self.trim_x + initial_perturb
        self.saturation = self.system.saturation

        self.poly_phi = PolynomialFeatures(degree=phi_deg, include_bias=False)

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
        """output : trimmed x"""
        return self.system.state - self.trim_x

    def step(self, theta):
        done = self.clock.time_over()
        time = self.clock.get()
        x_trimmed = self.observation()
        IR = self.IR_system.state
        u_trimmed = self.get_behavior(theta, x_trimmed[None, :])[0]

        if np.any(np.abs(x_trimmed + self.trim_x)[(1, 3), ] > np.deg2rad(30)):
            done = True

        self.update(theta)

        next_x_trimmed = self.observation()
        nIR = self.IR_system.state
        reward = nIR - IR

        next_u_trimmed = self.get_behavior(theta, next_x_trimmed[None, :])[0]

        info = {
            "time": time,
            "state": x_trimmed,
            "action": u_trimmed,
            "next_action": next_u_trimmed,
            "reward": reward,
            "next_state": next_x_trimmed
        }
        return next_x_trimmed, reward, done, info

    def set_dot(self, time, theta):
        x, _ = self.observe_list()
        x_trimmed = x - self.trim_x
        u_trimmed = self.get_behavior(theta, x_trimmed[None, :])[0]
        u = u_trimmed + self.trim_u

        self.system.dot = self.system.deriv(x, u)
        self.systems_dict["IR"].dot = self.reward(x_trimmed, u_trimmed)

    def reward(self, x_trimmed, u_trimmed):
        return (x_trimmed.dot(self.Q).dot(x_trimmed)
                + u_trimmed.dot(self.R).dot(u_trimmed))

    def phi(self, x_trimmed):
        poly = self.poly_phi.fit_transform(np.atleast_2d(x_trimmed))
        return poly

    def get_behavior(self, theta, x_trimmed):
        """output: trimmed and saturated u"""
        assert np.ndim(x_trimmed) == 2, "dim(x) = (batch, size_x)"
        u_trimmed = self.phi(x_trimmed).dot(theta)
        # return self.saturation(self.trim_u + u_trimmed) - self.trim_u
        return u_trimmed


class FixedParamEnv(BaseEnv):
    """This Env takes the learnt policy when initialized,
    then simulates with the fixed policy."""
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def step(self):
        time = self.clock.get()
        x, = self.observe_list()
        u = self.get_behavior(self.theta, x_trimmed[None, :])[0] + self.u_trimmed
        super().step(self.theta)

        info = {
            "time": time,
            "state": x,
            "control": u,
            "reward": reward,
        }
        return info
