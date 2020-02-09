import click
import numpy as np
import tqdm
import os
import glob
import time
import functools
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.multiprocessing as mp

import fym.logging as logging

import envs
import agents
import gan
import utils


@click.group()
def main():
    pass


@main.command()
@click.option("--number", "-n", default=50)
@click.option("--log-dir", default="data/samples")
@click.option("--max-workers", "-m", default=None)
def sample(**kwargs):

    print("Sample trajectories ...")

    prog = functools.partial(_sample_prog, log_dir=kwargs["log_dir"])

    t0 = time.time()
    with ProcessPoolExecutor(kwargs["max_workers"]) as p:
        list(tqdm.tqdm(
            p.map(prog, range(kwargs["number"])),
            total=kwargs["number"]
        ))

    print(f"Elapsed time: {time.time() - t0:5.2f} sec"
          f" > Finished - saved in {kwargs['log_dir']}")


def _sample_prog(i, log_dir):
    env = envs.BaseEnv(
        initial_perturb=[0, 0, 0, 0.1],
        dt=0.01, max_t=20,
        solver="rk4", ode_step_len=1,
    )
    agent = agents.BaseAgent(env, theta_init=0)
    agent.add_noise(scale=0.03, tau=2)
    file_name = f"{i:03d}.h5"
    _sample(env, agent, log_dir, file_name)


def _sample(env, agent, log_dir, file_name):
    logger = logging.Logger(
        log_dir=log_dir, file_name=file_name, max_len=100)

    obs = env.reset("random")
    while True:
        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        logger.record(**info)
        obs = next_obs

        if done:
            break

    env.close()
    logger.close()


@main.command()
@click.argumet("sample", nargs=-1, type=click.Path())
@click.option("--all", "mode", flag_value="all", default=True)
@click.option("--gan", "mode", flag_value="gan")
@click.option("--copdac", "mode", flag_value="copdac")
@click.option("--gan-lr", default=2e-4)
@click.option("--out", "-o", "savepath", default="data/trained.h5")
@click.option("--max-epoch", "-n", "max_epoch", default=1500)
@click.option("--batch-size", "-b", default=64)
def train(sample, **kwargs):
    samplefiles = utils.parse_file(sample, ext=".h5")

    if kwargs["mode"] == "gan" or kwargs["mode"] == "all":
        torch.manual_seed(0)
        np.random.seed(0)

        print("Train GAN ...")

        agent = gan.GAN(lr=kwargs["gan_lr"], x_size=4, u_size=4, z_size=100)

        t0 = time.time()
        for epoch in tqdm.trange(kwargs["max_epoch"]):
            dataloader = gan.get_dataloader(
                samplefiles, shuffle=True, batch_size=kwargs["batch_size"])

            loss_d = loss_g = 0
            for i, data in enumerate(dataloader):
                agent.set_input(data)
                agent.train()
                loss_d += agent.loss_d.mean().detach().numpy()
                loss_g += agent.loss_g.mean().detach().numpy()

        print(f"Elapsed time: {time.time() - t0:5.2f} sec")

    if kwargs["mode"] == "copdac" or kwargs["mode"] == "all":
        np.random.seed(4)

        env = envs.BaseEnv(
            initial_perturb=[0, 0, 0, 0.2])
        logger = logging.Logger(
            log_dir=".", file_name=kwargs["savepath"], max_len=100)

        agentlist = ("COPDAC", "RegCOPDAC")
        for agentname in agentlist:
            Agent = getattr(agents, agentname)
            agent = Agent(
                env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2, lrc=2e-3,
                w_init=0.03, v_init=0.03, theta_init=0,
                maxlen=100, batch_size=64
            )
            print(f"Training {agentname}...")
            _train_on_samples(env, agent, logger, **kwargs)

        logger.close()


def _train_on_samples(env, agent, logger, **kwargs):
    from collections import deque

    expname = "-".join([type(n).__name__ for n in (env, agent)])

    data = logging.load(kwargs["sample_dir"])
    data_list = [data[k] for k in ("state", "action", "reward", "next_state")]
    agent.buffer = deque([d for d in zip(*data_list)])

    recording_freq = int(kwargs["max_epoch"] / 200)
    for epoch in tqdm.trange(kwargs["max_epoch"]):
        agent.train()

        if (np.any(np.isnan(agent.w)) or np.any(np.isnan(agent.theta))
                or np.any(np.isnan(agent.v))):
            break

        if epoch % recording_freq == 0 or epoch == kwargs["max_epoch"]:
            logger.record(**{
                expname: dict(
                    epoch=epoch,
                    w=agent.w,
                    v=agent.v,
                    theta=agent.theta,
                    delta=agent.delta
                )
            })


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
            initial_perturb=[1, 0.0, 0, np.deg2rad(10)],
            dt=0.01, max_t=40, solver="rk4",
            ode_step_len=1, odeint_option={}
        )
        agent = getattr(agents, agentname)(
            env, lrw=1e-2, lrv=1e-2, lrtheta=1e-2, lrc=2e-3,
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
