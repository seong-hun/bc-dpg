import functools
import itertools
import numpy as np
from scipy.interpolate import interp1d

import fym
from fym.core import BaseEnv, BaseSystem
import fym.agents.LQR
import fym.logging as logging
from fym.models.aircraft import MorphingLon
from fym.utils.linearization import jacob_analytic


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

        self.set_delay(self.systems, T)

        trim_x, trim_u = self.system.get_trim()
        self.trim_x = trim_x
        self.trim_u = trim_u

        x = self.system.state
        phi_Q = self.phi_Q(x, np.zeros(4))
        phi_u = self.phi_u(x)
        self.append_systems({
            "WQ": BaseSystem(np.zeros_like(phi_Q), name="Q weights"),
            "Wu": BaseSystem(
                np.zeros(phi_u.shape + trim_u.shape),
                name="u weights"
            )
        })

        self.grad_u_phi_Q = jacob_analytic(self.phi_Q, i=1)

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

        if self.delay.available():
            self.delay.set_states(time)
            d_state = self.system.d_state
            d_Wu = self.systems_dict["Wu"].d_state
            d_control = self.get_control(d_Wu, d_state)
        else:
            d_state = np.zeros_like(state)
            d_control = np.zeros_like(control)

        self.update(action)
        info = {
            "time": time,
            "state": state,
            "control": control,
            "delayed_state": d_state,
            "delayed_control": d_control,
        }

        return self.observation(), 0, done, info

    def set_dot(self, time, action):
        if self.delay.available():
            self.delay.set_states(time)

        x = self.system.state
        Wu = self.systems_dict["Wu"].state
        u = self.get_control(Wu, x)  # delt, dele, eta1, eta2
        tx = x - self.trim_x
        tu = u - self.trim_u

        # There are four states in the system.
        # Each state needs to be set the derivative.
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

            self.systems_dict["dr"].dot = self.reward(dx, du)
            self.systems_dict["WQ"].dot = - 1e-2 * eQ * del_phi
            self.systems_dict["Wu"].dot = - 1e-2 * grad_Q
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

    def reward(self, x, u):
        return 1e-3 * (x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))


def get_poly(p, deg=2):
    p = np.hstack((1, p))
    return np.array([
        functools.reduce(lambda a, b: a * b, tup)
        for tup in itertools.combinations_with_replacement(p, deg)
    ])


def run(env):
    logger = logging.Logger(log_dir="data", file_name="tmp.h5")
    env.reset()
    while True:
        env.render()

        action = 0
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        if done:
            break

    env.close()
    logger.close()
    return logger.path


def plot(savepath):
    import matplotlib.pyplot as plt

    data = logging.load(savepath)

    canvas = []
    fig, axes = plt.subplots(2, 2, sharex=True)
    for ax in axes.flat:
        ax.mod = 1
    axes[0, 0].set_ylabel(r"$v$ [m/s]")
    axes[0, 1].set_ylabel(r"$\alpha$ [deg]")
    axes[0, 1].mod = np.rad2deg(1)
    axes[1, 0].set_ylabel(r"$q$ [deg/s]")
    axes[1, 0].mod = np.rad2deg(1)
    axes[1, 1].set_ylabel(r"$\gamma$ [deg]")
    axes[1, 1].mod = np.rad2deg(1)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    canvas.append((fig, axes))

    fig, axes = plt.subplots(2, 2, sharex=True)
    for ax in axes.flat:
        ax.mod = 1
    axes[0, 0].set_ylabel(r"$\delta_t$")
    axes[0, 1].set_ylabel(r"$\delta_e$ [deg]")
    axes[0, 1].mod = np.rad2deg(1)
    axes[1, 0].set_ylabel(r"$\eta_1$")
    axes[1, 1].set_ylabel(r"$\eta_2$")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    canvas.append((fig, axes))

    time = data["time"]

    axes = canvas[0][1]
    for ax, x in zip(axes.flat, data["state"].T):
        ax.plot(time, x * ax.mod, color="k")
    # axes[0, 0].lines[0].set_label("True")
    # axes[0, 0].legend(*axes[0].get_legend_handles_labels())

    axes = canvas[1][1]
    for ax, u in zip(axes.flat, data["control"].T):
        ax.plot(time, u * ax.mod, color="k")
    # axes[0, 0].lines[0].set_label("True")
    # axes[0, 0].legend(*axes[0].get_legend_handles_labels())

    for fa in canvas:
        fa[0].tight_layout()

    plt.show()


if __name__ == "__main__":
    T = 0.5
    env = Env([16, 0, 0, 0], T, dt=0.1, max_t=10, ode_step_len=4)
    env.reset()
    savepath = run(env)
    plot(savepath)
    # plot("data/tmp.h5")
