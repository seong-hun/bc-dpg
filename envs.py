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
    Q = np.diag([1, 100, 10, 100])
    R = np.diag([10, 10, 1, 1])

    def __init__(self, initial_state, T, **kwargs):
        self.system = MorphingLon(initial_state)
        super().__init__(
            systems_dict={
                "main": self.system,
                "r": BaseSystem(0, name="Integral reward"),
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
                0.03 * 2 * (np.random.random(phi_Q.shape) - 0.5),
                name="Q weights"
            ),
            "Wu": BaseSystem(
                0.0 * (np.random.random(phi_u.shape + trim_u.shape) - 0.5),
                name="u weights"
            )
        })

        self.grad_u_phi_Q = jacob_analytic(self.phi_Q, i=1)

        self.set_delay(
            (self.system, self.systems_dict["Wu"], self.systems_dict["r"]),
            T
        )

    def reset(self):
        super().reset()
        self.system.state = self.trim_x + [0, 0, 0, 0.02]
        return self.observation()

    def observe_d_list(self):
        return [system.d_state for system in self.delay.systems]

    def observation(self):
        if self.delay.available():
            x, r, WQ, Wu = self.observe_list()

            tx = x - self.trim_x

            time = self.clock.get()
            self.delay.set_states(time)

            dx, dWu, dr = self.observe_d_list()
            du = self.get_behavior(dWu, dx, time)

            dtx = dx - self.trim_x
            dtu = du - self.trim_u

            us = self.get_target(WQ, Wu, x)
            tus = us - self.trim_u

            del_phi = self.phi_Q(tx, tus) - self.phi_Q(dtx, dtu)
            integral_reward = r - dr
            return del_phi, integral_reward
        else:
            return None

    def step(self, action):
        done = self.clock.time_over()
        time = self.clock.get()
        x, _, WQ, Wu = self.observe_list()
        bu = self.get_behavior(Wu, x, time)
        reward = self.reward(x - self.trim_x, bu - self.trim_u)

        F, G = action or (0, 0)
        td_error = nla.norm(np.dot(F, WQ) + G)

        if np.any(np.abs(x[(1, 3), ]) > np.deg2rad(30)):
            done = True

        self.update(action)

        info = {
            "time": time,
            "trim_x": self.trim_x,
            "trim_u": self.trim_u,
            "state": x,
            "control": bu,
            "Wu": Wu.ravel(),
            "WQ": WQ.ravel(),
            "reward": reward,
            "td_error": td_error,
        }

        return self.observation(), reward, done, info

    def get_behavior(self, Wu, x, t,
                     noise=[0.05, 0.02, 0.05, 0.05], tau=5):
        """If ``x = self.trim_x``, then ``del_u = 0``. This is ensured by
        the structure of ``phi_u`` which has no constant term. Also,
        the behavior policy should always be saturated by the control limits
        defined by the system."""
        del_u = (Wu.T.dot(self.phi_u(x - self.trim_x))
                 + 0 * np.exp(-t / tau) * np.array(noise) * np.random.randn())
        return self.system.saturation(self.trim_u + del_u)

    def get_target(self, WQ, Wu, x):
        del_u = Wu.T.dot(self.phi_u(x - self.trim_x))
        return self.system.saturation(self.trim_u + del_u)

    def set_dot(self, time, action):
        x, r, WQ, Wu = self.observe_list()
        bu = self.get_behavior(Wu, x, time)

        tx = x - self.trim_x
        reward = self.reward(tx, bu - self.trim_u)

        if self.delay.available():
            self.delay.set_states(time)

            # ``action`` is not ``None`` Only after the delay is available
            # and there are enough data in ``buffer`` of the agent.
            F, G = action or (0, 0)

            dx, dWu, dr = self.observe_d_list()
            dbu = self.get_behavior(dWu, dx, time)
            dtx = dx - self.trim_x
            dtbu = dbu - self.trim_u

            integral_reward = r - dr

            phi_u = self.phi_u(tx)
            us = self.get_target(WQ, Wu, x)
            tus = us - self.trim_u

            del_phi = self.phi_Q(tx, tus) - self.phi_Q(dtx, dtbu)
            eQ = WQ.dot(del_phi) + integral_reward

            grad_u_phi_Q = self.grad_u_phi_Q(tx, tus)
            grad_Q = np.outer(phi_u, grad_u_phi_Q.T.dot(WQ))

            self.systems_dict["WQ"].dot = - 10 * (
                eQ * del_phi
                + 100 * (np.dot(F, WQ) + G)
            )
            self.systems_dict["Wu"].dot = - 10 * grad_Q
        else:
            self.systems_dict["WQ"].dot = np.zeros_like(
                self.systems_dict["WQ"].state)
            self.systems_dict["Wu"].dot = np.zeros_like(
                self.systems_dict["Wu"].state)

        self.system.dot = self.system.deriv(x, bu)
        self.systems_dict["r"].dot = reward

    def phi_Q(self, x, u, deg=2):
        n = len(get_poly(u, deg=2))
        return get_poly(np.hstack((1, x, u)), deg=2)[1:-n]

    def phi_u(self, x, deg=[1, 2]):
        return get_poly(x, deg=deg)

    def reward(self, tx, tu):
        return tx.dot(self.Q).dot(tx) + tu.dot(self.R).dot(tu)


def get_poly(p, deg=2):
    if isinstance(deg, int):
        if deg == 0:
            return 1
        else:
            return np.array([
                functools.reduce(lambda a, b: a * b, tup)
                for tup in itertools.combinations_with_replacement(p, deg)
            ])
    elif isinstance(deg, list):
        return np.hstack([get_poly(p, deg=d) for d in deg])
    else:
        raise ValueError("deg should be an integer or a list of integers.")
