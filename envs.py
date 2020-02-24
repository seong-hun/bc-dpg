import numpy as np
import numpy.linalg as nla
import numba as nb

from sklearn.preprocessing import PolynomialFeatures

import fym
import fym.core
import fym.agents.LQR
import fym.logging as logging
from fym.models.aircraft import MorphingLon
from fym.utils.linearization import jacob_analytic

from utils import get_poly
import common


@nb.njit()
def quad_reward(x_trimmed, u_trimmed, Q, R):
    return x_trimmed.dot(Q).dot(x_trimmed) + u_trimmed.dot(R).dot(u_trimmed)


class BaseEnv(fym.core.BaseEnv):
    """This Env takes the learnt policy when initialized,
    then simulates with the fixed policy."""
    Q = np.diag([1, 100, 10, 100]).astype(np.float)
    R = np.diag([50, 1, 1, 1]).astype(np.float)

    def __init__(self, initial_perturb, **kwargs):
        self.system = MorphingLon(initial_perturb)
        self.IR_system = fym.core.BaseSystem(0, name="integral reward")
        super().__init__(
            systems_dict={
                "main": self.system,
                "IR": self.IR_system,
            },
            name="MorphingLon",
            eager_stop=self.stop_cond,
            **kwargs
        )
        self.initial_perturb = initial_perturb
        self.trim_x, self.trim_u = self.system.get_trim()
        self.system.initial_state = self.trim_x + initial_perturb
        self.saturation = self.system.saturation

    def stop_cond(self, t_hist, ode_hist):
        index = np.where(
            np.any(np.abs(ode_hist[:, (1, 3)]) > np.deg2rad(30), axis=1))[0]
        if index.size == 0:
            return t_hist, ode_hist, False
        else:
            return t_hist[:index[0] + 1], ode_hist[:index[0] + 1], True

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


class FixedParamEnv(BaseEnv):
    def __init__(self, initial_perturb, **kwargs):
        super().__init__(
            initial_perturb,
            **kwargs,
            logger_callback=self.get_info,
        )

    def get_action(self, t, x):
        return self.trim_u

    def set_inner_ctrl(self, inner_ctrl):
        inner_ctrl.set_trim(self.trim_x, self.trim_u)
        self.get_action = self.wrap_action(
            inner_ctrl.get, self.system.saturation)

    def get_info(self, i, t, y, t_hist, ode_hist):
        ny = ode_hist[i + 1]
        x, IR, nx, nIR = [
            p[system.flat_index].reshape(system.state_shape)
            for p in (y, ny)
            for system in self.systems
        ]
        u = self.get_action(t, x)
        reward = nIR - IR
        return {
            "time": t,
            "state": x,
            "action": u,
            "reward": reward,
            "next_state": nx
        }

    def wrap_action(self, get_action, saturation):
        def wrap(*args, **kwargs):
            return saturation(get_action(*args, **kwargs))
        return wrap

    def reset(self, mode=None):
        super().reset()
        if mode == "random":
            self.system.state = (
                self.trim_x
                + self.initial_perturb
                + [1, 0.05, 0.05, 0.05] * np.random.randn(4)
            )

    def step(self):
        done = self.clock.time_over()
        _, ode_hist, eager_done = self.update()
        done = done or eager_done
        return None, None, done, {}

    def set_dot(self, time):
        x, _ = self.observe_list()
        u = self.get_action(time, x)

        self.system.dot = self.system.deriv(x, u)
        self.systems_dict["IR"].dot = self.reward(x, u)

    def reward(self, x, u):
        x_trimmed = x - self.trim_x
        u_trimmed = u - self.trim_u
        return quad_reward(x_trimmed, u_trimmed, self.Q, self.R)


if __name__ == "__main__":
    import agents

    behavior = agents.Linear(xdim=4, udim=4)
    behavior.set_param(0)
    ou_noise = common.OuNoise(0, 0.01, dt=1, max_t=100, decay=30)
    behavior.add_noise(ou_noise)
    env = FixedParamEnv(
        initial_perturb=[0, 0, 0, 0.1], inner_ctrl=behavior,
        max_t=100, solver="odeint", dt=40, ode_step_len=1000,
        logging_off=False,
    )

    env.reset("random")
    while True:
        env.render()
        _, _, done, _ = env.step()

        if done:
            break

    env.close()
