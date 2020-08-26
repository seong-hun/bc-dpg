import numpy as np
import numpy.linalg as nla
import random

import fym
from fym.core import BaseEnv, BaseSystem
import fym.agents.LQR
import fym.logging as logging
import fym.models.aircraft as aircraft
from fym.utils.linearization import jacob_analytic

from utils import get_poly
import args

np.random.seed(1)


# class BaseEnv(BaseEnv):
#     Q = np.diag([1, 100, 10, 100])
#     R = np.diag([50, 1, 1, 1])

#     def __init__(self, initial_perturb, W_init, **kwargs):
#         super().__init__()
#         self.system = MorphingLon()
#         self.IR_system = BaseSystem(0, name="integral reward")

#         self.initial_perturb = initial_perturb
#         trim_x, trim_u = self.system.get_trim()
#         self.trim_x = trim_x
#         self.trim_u = trim_u
#         self.system.initial_state = trim_x + initial_perturb

#     def reset(self, initial_perturb=None):
#         if initial_perturb == "random":
#             self.system.initial_state = (
#                 self.trim_x
#                 + self.initial_perturb
#                 + [1, 0.05, 0.05, 0.05] * np.random.randn(4)
#             )

#         super().reset()
#         return self.observation()

#     def observation(self):
#         x = self.system.state - self.trim_x
#         IR = self.IR_system.state
#         return x, IR

#     def step(self, action):
#         time = self.clock.get()
#         x, IR = self.observation()
#         u = action + self.trim_u

#         if np.any(np.abs(x[(1, 3), ]) > np.deg2rad(30)):
#             done = True

#         done, *_ = self.update(u)

#         next_x = self.observation()
#         nIR = self.IR_system.state
#         reward = nIR - IR

#         info = {
#             "time": time,
#             "state": x,
#             "action": action,
#             "reward": reward,
#             "next_state": next_x
#         }

#         return next_x, done, info

#     def set_dot(self, time, u):
#         x, _ = self.observe_list()

#         self.system.dot = self.system.deriv(x, u)
#         self.systems_dict["IR"].dot = self.reward(x, u)

#     def reward(self, x, u):
#         tx = x - self.trim_x
#         tu = u - self.trim_u
#         return tx.dot(self.Q).dot(tx) + tu.dot(self.R).dot(tu)


class MainSystem(aircraft.MorphingLon):
    def __init__(self):
        super().__init__()
        trim_x, trim_delta, trim_eta = self.get_trim()
        self.trim_x = trim_x
        self.trim_u = np.vstack((trim_delta, trim_eta))

    def get_trimmed(self, x, u):
        return x - self.trim_x, u - self.trim_u


class IRSystem(BaseSystem):
    Q = np.diag([1, 100, 10, 100])
    R = np.diag([50, 1, 1, 1])

    def get_r(self, tx, tu):
        return tx.T.dot(self.Q).dot(tx) + tu.T.dot(self.R).dot(tu)


class SampleEnv(BaseEnv):
    # behavior_types = ["Fourier", "parameterized policy",
    #                   "pure random", "fixed"]
    behavior_types = ["Fourier"]

    def __init__(self):
        super().__init__(dt=args.SAMPLE.TIMESTEP, max_t=args.SAMPLE.FINALTIME)
        self.system = MainSystem()
        self.IR_system = IRSystem()

    def reset(self):
        super().reset()
        perturb = args.SAMPLE.PERTURB * np.random.randn(4, 1)
        self.system.initial_state = self.system.trim_x + perturb

        # Randomness for the behavior policy
        self.btype = random.choice(self.behavior_types)
        self.bparam = np.random.rand(10)

    def observation(self):
        obs = self.system.state - self.system.trim_x
        return obs

    def step(self):
        t = self.clock.get()
        x = self.system.state
        IR = self.IR_system.state
        u = self.get_u(t, x)
        state, action = self.system.get_trimmed(x, u)

        *_, done = self.update()
        next_state = self.system.state - self.system.trim_x
        reward = self.IR_system.state - IR

        info = dict(
            time=t,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        return done, info

    def set_dot(self, t):
        x = self.system.state
        u = self.get_u(t, x)
        tx, tu = self.system.get_trimmed(x, u)

        self.system.dot = self.system.deriv(x, u[:2], u[2:])
        self.IR_system.dot = self.IR_system.get_r(tx, tu)

    def get_u(self, t, x):
        u = 0

        if self.btype == "Fourier":
            for i, w in enumerate(self.bparam, 1):
                noise = np.sin(i * w * t + w**2/1.1)
                if i * w > 3:
                    noise = noise * np.exp(-0.1 * i * w * t)
                u = u + noise
            u = u / len(self.bparam)
        elif self.btype == "parameterized policy":
            pass
        elif self.btype == "pure random":
            pass
        elif self.btype == "fixed":
            u = u + np.vstack(self.bparam[:4])

        u = u * np.exp(-0.1 * t)
        u = u * np.vstack((0.05, np.deg2rad(2), 0.05, 0.05))
        return u + self.system.trim_u


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
