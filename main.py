import numpy as np

import fym
import fym.core
import fym.agents.LQR


class System(fym.core.BaseSystem):
    A = np.array([
        [-1.01887, 0.90506, -0.00215],
        [0.82225, -1.07741, -0.17555],
        [0, 0, -1]
    ])
    B = np.array([
        [0],
        [0],
        [1]
    ])

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u)


class Env(fym.core.BaseEnv):
    def __init__(self):
        self.system = System([1, -1, 0])

        super().__init__(
            systems_dict={"main": self.system},
            dt=0.1,
            max_t=10,
            ode_step_len=4,
        )

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        return self.observe_flat()

    def step(self, action):
        done = self.clock.time_over()
        self.update(action)
        return self.observation(), 0, done, {}

    def derivs(self, time, action):
        self.system.set_dot(self.system.deriv(self.system.state, action))


class Lqr:
    Q = np.diag([10, 10, 1])
    R = np.diag([1])

    def __init__(self, env):
        self.K, *_ = fym.agents.LQR.clqr(
            env.system.A, env.system.B, self.Q, self.R
        )

    def get_action(self, obs):
        return - self.K.dot(obs)


def run(env, agent):
    obs = env.reset()
    while True:
        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        print(obs, action)

        if done:
            break

        obs = next_obs


env = Env()
agent = Lqr(env)
run(env, agent)

# data = generate_samples()
