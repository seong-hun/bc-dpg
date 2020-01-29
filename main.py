import click
import numpy as np
import tqdm

import fym.logging as logging

import envs
import agents


@click.group()
def main():
    pass


@main.command()
@click.option("--weights", "-w", default="data/weights.h5")
@click.option("--out", "-o", default="data/run.h5")
@click.option("--with-plot", "-p", is_flag=True)
def run(**kwargs):
    env = envs.Env(
        initial_perturb=[0, 0, 0, 0.1], W_init=0.0,
        dt=0.01, max_t=40, solver="rk4",
        ode_step_len=1, odeint_option={},
    )
    agent = agents.Agent(
        env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2,
        w_init=0.03, v_init=0.03, theta_init=0,
        maxlen=100, batch_size=16
    )
    agent.load_weights(kwargs["weights"])

    _run(env, agent, **kwargs)

    if kwargs["with_plot"]:
        import figures

        figures.plot(kwargs["out"])
        figures.show()


def _run(env, agent, **kwargs):
    logger = logging.Logger(
        log_dir=".", file_name=kwargs["out"], max_len=100)
    obs = env.reset()
    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        # agent.append(obs, action, reward, next_obs)
        # agent.train()

        obs = next_obs

        if done:
            break

    env.close()
    logger.close()
    return logger.path


@main.command()
@click.option("--number", "-n", default=50)
@click.option("--out", "-o", default="sample.h5")
def sample(**kwargs):
    env = envs.Env(
        initial_perturb=[0, 0, 0, 0.1], W_init=0.0,
        dt=0.01, max_t=20, solver="rk4",
        ode_step_len=1, odeint_option={},
    )
    agent = agents.Agent(
        env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2,
        w_init=0.03, v_init=0.03, theta_init=0,
        maxlen=100, batch_size=16
    )

    print("Sample trajectories")
    logger = logging.Logger(
        log_dir="data", file_name=kwargs["out"], max_len=100
    )
    for i in tqdm.trange(kwargs["number"]):
        _sample(env, agent, logger)
    logger.close()


def _sample(env, agent, logger):
    obs = env.reset("random")
    while True:
        # env.render()

        action = agent.get_action(obs, noise_scale=0.03)
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        obs = next_obs

        if done:
            break

    env.close()


@main.command()
@click.option("--in", "-i", "datapath", default="data/sample.h5")
@click.option("--out", "-o", "savepath", default="data/trained.h5")
@click.option("--max-epoch", "-n", "max_epoch", default=1000)
def train(**kwargs):
    _train_on_samples(**kwargs)


def _train_on_samples(**kwargs):
    from collections import deque

    np.random.seed(0)
    data = logging.load(kwargs["datapath"])
    env = envs.Env(
        initial_perturb=[0, 0, 0, 0.2], W_init=0.0,
    )
    agent = agents.Agent(
        env, lrw=1e-2, lrv=1e-1, lrtheta=1e-2,
        w_init=0.00, v_init=0.00, theta_init=0.00,
        maxlen=100, batch_size=16
    )

    data_list = [data[k] for k in ("state", "action", "reward", "next_state")]
    agent.buffer = deque([d for d in zip(*data_list)])

    logger = logging.Logger(
        log_dir=".", file_name=kwargs["savepath"], max_len=100
    )
    recording_freq = int(kwargs["max_epoch"] / 200)
    for epoch in tqdm.trange(kwargs["max_epoch"]):
        agent.train()

        if (np.any(np.isnan(agent.w)) or np.any(np.isnan(agent.theta))
                or np.any(np.isnan(agent.v))):
            break

        # import ipdb; ipdb.set_trace()
        if epoch % recording_freq == 0 or epoch == kwargs["max_epoch"]:
            logger.record(
                epoch=epoch,
                w=agent.w,
                v=agent.v,
                theta=agent.theta
            )

    logger.close()
    logging.save("data/weights.h5",
                 {"w": agent.w, "v": agent.v, "theta": agent.theta})


@main.command()
@click.argument("path")
@click.option("--train", "-t", is_flag=True)
def plot(path, **kwargs):
    import figures

    if kwargs["train"]:
        figures.train_plot(path)
    else:
        figures.plot(path)

    figures.show()


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
