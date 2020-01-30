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
@click.option("--trained", "-w", default="data/trained.h5")
@click.option("--out", "-o", default="data/run.h5")
@click.option("--with-plot", "-p", is_flag=True)
def run(**kwargs):
    logger = logging.Logger(
        log_dir=".", file_name=kwargs["out"], max_len=100)
    dataset = logging.load(kwargs["trained"])
    for expname, data in dataset.items():
        envname, agentname = expname.split("-")
        env = getattr(envs, envname)(
            initial_perturb=[1, 0.0, 0, np.deg2rad(10)], W_init=0.0,
            dt=0.01, max_t=40, solver="rk4",
            ode_step_len=1, odeint_option={}
        )
        agent = getattr(agents, agentname)(
            env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2,
            w_init=0.03, v_init=0.03, theta_init=0,
            maxlen=100, batch_size=16
        )
        agent.load_weights(data)

        print(f"Runnning {expname} ...")
        _run(env, agent, logger, expname, **kwargs)

    logger.close()

    if kwargs["with_plot"]:
        import figures

        dataset = logging.load(kwargs["out"])
        figures.plot_mult(dataset)

        figures.show()


def _run(env, agent, logger, expname, **kwargs):
    obs = env.reset()
    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        logger.record(**{expname: info})

        obs = next_obs

        if done:
            break

    env.close()
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
# @click.option("--agents", "-a", "agentlist", multiple=True, )
def train(**kwargs):
    np.random.seed(1)
    env = envs.BaseEnv(
        initial_perturb=[0, 0, 0, 0.2], W_init=0.0)
    logger = logging.Logger(
        log_dir=".", file_name=kwargs["savepath"], max_len=100)

    agentlist = ("COPDAC", "RegCOPDAC")
    for agentname in agentlist:
        Agent = getattr(agents, agentname)
        agent = Agent(
            env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2,
            w_init=0.03, v_init=0.03, theta_init=0,
            maxlen=100, batch_size=64
        )
        print(f"Training {agentname}...")
        _train_on_samples(env, agent, logger, **kwargs)

    logger.close()


def _train_on_samples(env, agent, logger, **kwargs):
    from collections import deque

    expname = "-".join([type(n).__name__ for n in (env, agent)])

    data = logging.load(kwargs["datapath"])
    data_list = [data[k] for k in ("state", "action", "reward", "next_state")]
    agent.buffer = deque([d for d in zip(*data_list)])

    recording_freq = int(kwargs["max_epoch"] / 200)
    for epoch in tqdm.trange(kwargs["max_epoch"]):
        agent.train()

        if (np.any(np.isnan(agent.w)) or np.any(np.isnan(agent.theta))
                or np.any(np.isnan(agent.v))):
            break

        # import ipdb; ipdb.set_trace()
        if epoch % recording_freq == 0 or epoch == kwargs["max_epoch"]:
            logger.record(**{
                expname: dict(
                    epoch=epoch, w=agent.w, v=agent.v, theta=agent.theta)
            })


@main.command()
@click.argument("path")
@click.option("--train", "-t", is_flag=True)
def plot(path, **kwargs):
    import figures

    if kwargs["train"]:
        figures.train_plot(path)
    else:
        dataset = logging.load(path)
        figures.plot_mult(dataset)

    figures.show()


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
