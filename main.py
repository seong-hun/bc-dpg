import click

import fym.logging as logging

import envs
import agents


@click.group()
def main():
    pass


@main.command()
def run(**kwargs):
    env = envs.Env(
        initial_state=[16, 0, 0, 0], W_init=0.0, eta=18,
        dt=0.01, max_t=20, solver="rk4",
        ode_step_len=1, odeint_option={},
    )
    agent = agents.Agent(env, lr=1e-2, Wc_init=0.03, maxlen=100, batch_size=16)
    _run(env, agent)


def _run(env, agent):
    logger = logging.Logger(
        log_dir="data", file_name="tmp.h5", max_len=100)
    obs = env.reset()
    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        agent.append(obs, action, reward, next_obs)
        agent.train()

        obs = next_obs

        if done:
            break

    env.close()
    logger.close()
    return logger.path


@main.command()
@click.argument("path")
def plot(**kwargs):
    import figures

    figures.plot(kwargs["path"])


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
