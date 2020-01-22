import functools
import itertools

import numpy as np
import numpy.linalg as nla

import fym
from fym.core import BaseEnv, BaseSystem
import fym.agents.LQR
import fym.logging as logging
from fym.models.aircraft import MorphingLon
from fym.utils.linearization import jacob_analytic

np.random.seed(1)


class Env(BaseEnv):
    Q = np.diag([10, 1, 100, 1])
    R = np.diag([100, 1, 10, 10])

    def __init__(self, initial_state, T, **kwargs):
        self.system = MorphingLon(initial_state)
        super().__init__(
            systems_dict={
                "main": self.system,
                "r": BaseSystem(0, name="Integral reward"),
                "dr": BaseSystem(0, name="Delayed integral reward"),
            },
            **kwargs
        )

        trim_x, trim_u = self.system.get_trim()
        self.trim_x = trim_x
        self.trim_u = trim_u

        x = self.system.state
        phi_Q = self.phi_Q(x, np.zeros(4))
        phi_u = self.phi_u(x)
        self.append_systems({
            "WQ": BaseSystem(
                0.05 * (np.random.random(phi_Q.shape) - 0.5),
                name="Q weights"
            ),
            "Wu": BaseSystem(
                0.05 * (np.random.random(phi_u.shape + trim_u.shape) - 0.5),
                name="u weights"
            )
        })

        self.grad_u_phi_Q = jacob_analytic(self.phi_Q, i=1)

        self.set_delay((self.system, self.systems_dict["Wu"]), T)

    def reset(self):
        super().reset()
        self.system.state = self.trim_x + [2, 0.1, 0, 0]
        return self.observation()

    def observation(self):
        return self.observe_flat()

    def step(self, action):
        done = self.clock.time_over()
        time = self.clock.get()
        state = self.system.state
        Wu = self.systems_dict["Wu"].state
        control = self.get_control(Wu, state)

        self.update(action)

        info = {
            "time": time,
            "state": state,
            "control": control,
        }

        return self.observation(), 0, done, info

    def set_dot(self, time, action):
        x = self.system.state
        Wu = self.systems_dict["Wu"].state
        u = self.get_control(Wu, x)  # delt, dele, eta1, eta2
        tx = x - self.trim_x
        tu = u - self.trim_u

        u = self.system.saturation(u)

        self.system.dot = self.system.deriv(x, u)
        self.systems_dict["r"].dot = self.reward(tx, tu)

        if self.delay.available():
            self.delay.set_states(time)

            dx = self.system.d_state
            dWu = self.systems_dict["Wu"].d_state
            du = self.get_control(dWu, dx)
            dtx = dx - self.trim_x
            dtu = du - self.trim_u

            integral_reward = (
                self.systems_dict["r"].state
                - self.systems_dict["dr"].state
            )

            WQ = self.systems_dict["WQ"].state
            phi_u = self.phi_u(tx)
            tus = tu

            del_phi = self.phi_Q(tx, tus) - self.phi_Q(dtx, dtu)
            eQ = WQ.dot(del_phi) + integral_reward
            grad_u_phi_Q = self.grad_u_phi_Q(tx, tu)
            grad_Q = np.outer(phi_u, grad_u_phi_Q.T.dot(WQ))

            self.systems_dict["dr"].dot = self.reward(dtx, dtu)
            self.systems_dict["WQ"].dot = - 5e-3 * eQ * del_phi
            self.systems_dict["Wu"].dot = - 5e-3 * grad_Q

            # print(np.abs(eQ * del_phi).max(), np.abs(grad_Q).max())
        else:
            self.systems_dict["dr"].dot = 0
            self.systems_dict["WQ"].dot = np.zeros_like(
                self.systems_dict["WQ"].state)
            self.systems_dict["Wu"].dot = np.zeros_like(
                self.systems_dict["Wu"].state)

    def get_control(self, Wu, x):
        return self.trim_u + Wu.T.dot(self.phi_u(x - self.trim_x))

    def phi_Q(self, x, u, deg=2):
        return get_poly(np.hstack((x, u)), deg=deg)

    def phi_u(self, x, deg=2):
        return get_poly(x, deg=deg)

    def reward(self, tx, tu):
        return 1e-3 * (tx.dot(self.Q).dot(tx) + tu.dot(self.R).dot(tu))


def get_poly(p, deg=2):
    p = np.hstack((1, p))
    return np.array([
        functools.reduce(lambda a, b: a * b, tup)
        for tup in itertools.combinations_with_replacement(p, deg)
    ])
