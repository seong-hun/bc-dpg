import click

import fym.logging as logging

import envs


@click.group()
def main():
    pass


@main.command()
def run(**kwargs):
    T = 0.05
    env = envs.Env(
        [16, 0, 0, 0], T, dt=0.01, max_t=25,
        ode_step_len=4, odeint_option={},
    )
    env.reset()
    _run(env)


def _run(env):
    logger = logging.Logger(log_dir="data", file_name="tmp.h5",
                            max_len=10)
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


@main.command()
@click.argument("path")
def plot(**kwargs):
    import figures

    figures.plot(kwargs["path"])


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
