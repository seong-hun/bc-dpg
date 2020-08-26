import os
from datetime import datetime
import sys
import click
import numpy as np
import tqdm
import logging
import matplotlib.pyplot as plt

import fym.logging

import envs
import agents
import args


@click.group()
def main():
    if not os.path.isdir(args.LOG_DIR):
        os.makedirs(args.LOG_DIR)
    if not os.path.isdir(args.IMG_DIR):
        os.makedirs(args.IMG_DIR)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(args.LOG_DIR, f"{datetime.now():%Y%m%d}.log"),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger('matplotlib.font_manager').disabled = True


@main.command()
@click.option("--number", "-n", default=50)
def sample(**kwargs):
    logging.info("===== Sample trajectories =====")

    env = envs.SampleEnv()

    plotter = SamplePlotter(env)

    for idx_sample in tqdm.trange(kwargs["number"]):
        while True:
            logger = fym.logging.Logger(f"data/samples/{idx_sample:03d}.h5")

            env.reset()

            while True:
                # env.render()
                done, info = env.step()
                logger.record(**info)

                if done:
                    break

            env.close()
            logger.close()

            # Check
            data = fym.logging.load(logger.path)
            lns = plotter.plot(data, args.SAMPLE.LINE.TMP)

            while True:
                user_input = input(f"Wanna keep {logger.path}? [Y/n/q]")
                if user_input == "n":
                    keep = False
                elif user_input in "Yy":
                    keep = True
                elif user_input == "q":
                    return
                else:
                    continue
                break

            if keep:
                for ln in lns:
                    ln.set(**args.SAMPLE.LINE.STORED)
                break
            else:
                for ln in lns:
                    ln.remove()
                os.remove(logger.path)

    logging.info("Sampling ends")


@main.command()
def plot_samples(**kwargs):
    files = [
        os.path.join(path, filename)
        for path, dirs, files in os.walk("data/samples")
        for filename in files
        if filename.endswith(".h5")
    ]
    files.sort()

    env = envs.SampleEnv()
    plotter = SamplePlotter(env)

    for path in files:
        data = fym.logging.load(path)
        plotter.plot(
            data, dict(lw=1, c=np.random.uniform(0.2, 0.8) * np.ones(3)))

    plt.savefig(os.path.join(args.IMG_DIR, "samples.png"))


class SamplePlotter():
    def __init__(self, env):
        self.env = env

        self.fig, self.axes = plt.subplots(4, 2, sharex=True)

        for ax, ylabel in zip(self.axes.T.flat, args.SAMPLE.LABELS.Y):
            ax.set_ylabel(ylabel)

        for ax in self.axes[-1]:
            ax.set_xlabel(args.SAMPLE.LABELS.X)

        plt.show(block=False)

    def plot(self, data, ln_kwargs):
        time = data["time"]
        state = data["state"] + self.env.system.trim_x
        action = data["action"] + self.env.system.trim_u

        lns = []
        for i, axrow in enumerate(self.axes):
            ax1, ax2 = axrow
            y1 = state[:, i, 0]
            y2 = action[:, i, 0]

            if i > 0:
                y1 = np.rad2deg(y1)

            if i == 1:
                y2 = np.rad2deg(y2)

            lns += ax1.plot(time, y1, **ln_kwargs)
            lns += ax2.plot(time, y2, **ln_kwargs)

        self.fig.tight_layout()
        plt.draw()

        return lns


@main.command()
@click.option("--trained", "-w", default="data/trained.h5")
@click.option("--out", "-o", default="data/run.h5")
@click.option("--with-plot", "-p", is_flag=True)
def run(**kwargs):
    logger = fym.logging.Logger(kwargs["out"])
    dataset = fym.logging.load(kwargs["trained"])
    for expname, data in dataset.items():
        envname, agentname = expname.split("-")
        env = getattr(envs, envname)(
            initial_perturb=[1, 0.0, 0, np.deg2rad(10)], W_init=0.0,
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

        dataset = fym.logging.load(kwargs["out"])
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
@click.option("--in", "-i", "datapath", default="data/sample.h5")
@click.option("--out", "-o", "savepath", default="data/trained.h5")
@click.option("--max-epoch", "-n", "max_epoch", default=1500)
# @click.option("--agents", "-a", "agentlist", multiple=True, )
def train(**kwargs):
    np.random.seed(4)
    env = envs.BaseEnv(
        initial_perturb=[0, 0, 0, 0.2], W_init=0.0)
    logger = fym.logging.Logger(
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

    data = fym.logging.load(kwargs["datapath"])
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
                    epoch=epoch,
                    w=agent.w,
                    v=agent.v,
                    theta=agent.theta,
                    delta=agent.delta
                )
            })


@main.command()
@click.argument("path")
@click.option("--train", "-t", is_flag=True)
def plot(path, **kwargs):
    import figures

    if kwargs["train"]:
        figures.train_plot(path)
    else:
        dataset = fym.logging.load(path)
        figures.plot_mult(dataset)

    figures.show()


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
