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

PARAMS = {
    "GAN": {
        "x_size": 4,
        "u_size": 4,
        "z_size": 100,
        "lr": 2e-4,
    },
    "COPDAC": {
        "lrw": 1e-2,
        "lrv": 1e-2,
        "lrtheta": 1e-2,
        "lrc": 2e-3,
        "lrg": 8e-1,
        "w_init": 0.03,
        "v_init": 0.03,
        "theta_init": 0,
        "maxlen": 100,
        "batch_size": 64,
    },
}


@click.group()
def main():
    pass


@main.command()
@click.option("--number", "-n", default=50)
@click.option("--log-dir", default="data/samples")
@click.option("--max-workers", "-m", default=None)
def sample(**kwargs):
    max_workers = int(kwargs["max_workers"] or os.cpu_count())
    assert max_workers <= os.cpu_count(), \
        f"workers should be less than {os.cpu_count()}"
    print(f"Sample trajectories with {max_workers} workers ...")

    prog = functools.partial(_sample_prog, log_dir=kwargs["log_dir"])

    t0 = time.time()
    with ProcessPoolExecutor(max_workers) as p:
        list(tqdm.tqdm(
            p.map(prog, range(kwargs["number"])),
            total=kwargs["number"]
        ))

    print(f"Elapsed time: {time.time() - t0:5.2f} sec"
          f" > saved in \"{kwargs['log_dir']}\"")


def _sample_prog(i, log_dir):
    np.random.seed(i)
    env = envs.BaseEnv(
        initial_perturb=[0, 0, 0, 0.1],
        dt=0.01, max_t=20,
        solver="rk4", ode_step_len=1,
    )
    agent = agents.BaseAgent(env, theta_init=0)
    agent.add_noise(scale=0.03, tau=2)
    file_name = f"{i:03d}.h5"

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
@click.argument("sample", nargs=-1, type=click.Path())
@click.option("--all", "mode", flag_value="all", default=True)
@click.option("--gan", "mode", flag_value="gan")
@click.option("--copdac", "mode", flag_value="copdac")
@click.option("--gan-lr", default=PARAMS["GAN"]["lr"])
@click.option("--use-cuda", is_flag=True)
@click.option("--gan-dir", default="data/gan")
@click.option("--copdac-dir", default="data/copdac")
@click.option("--continue", "-c", nargs=1, type=click.Path(exists=True))
@click.option("--max-epoch", "-n", default=100)
@click.option("--save-interval", "-s", default=10)
@click.option("--batch-size", "-b", default=64)
@click.option("--with-reg", is_flag=True)
@click.option("--with-gan", nargs=1, type=click.Path(exists=True))
@click.option("--out", "-o", "savepath", default="data/trained.h5")
def train(sample, mode, **kwargs):
    samplefiles = utils.parse_file(sample, ext="h5")

    if mode == "gan" or mode == "all":
        torch.manual_seed(0)
        np.random.seed(0)

        gandir = kwargs["gan_dir"]
        histpath = os.path.join(gandir, "train-history.h5")

        print("Train GAN ...")

        agent = gan.GAN(
            lr=kwargs["gan_lr"],
            x_size=PARAMS["GAN"]["x_size"],
            u_size=PARAMS["GAN"]["u_size"],
            z_size=PARAMS["GAN"]["z_size"],
            use_cuda=kwargs["use_cuda"],
        )

        if kwargs["continue"] is not None:
            epoch_start = agent.load(kwargs["continue"])
            logger = logging.Logger(
                path=histpath, max_len=kwargs["save_interval"], mode="r+")
        else:
            epoch_start = 0
            logger = logging.Logger(
                path=histpath, max_len=kwargs["save_interval"])

        t0 = time.time()
        for epoch in tqdm.trange(epoch_start,
                                 epoch_start + 1 + kwargs["max_epoch"]):
            dataloader = gan.get_dataloader(
                samplefiles, shuffle=True, batch_size=kwargs["batch_size"])

            loss_d = loss_g = 0
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                agent.set_input(data)
                agent.train()
                loss_d += agent.loss_d.mean().detach().numpy()
                loss_g += agent.loss_g.mean().detach().numpy()

            logger.record(epoch=epoch, loss_d=loss_d, loss_g=loss_g)

            if (epoch % kwargs["save_interval"] == 0
                    or epoch == epoch_start + 1 + kwargs["max_epoch"]):
                savepath = os.path.join(gandir, f"trained-{epoch:05d}.pth")
                agent.save(epoch, savepath)
                tqdm.tqdm.write(f"Weights are saved in {savepath}.")

        print(f"Elapsed time: {time.time() - t0:5.2f} sec")

    if mode == "copdac" or mode == "all":
        np.random.seed(1)

        env = envs.BaseEnv(initial_perturb=[0, 0, 0, 0.2])

        copdacdir = kwargs["copdac_dir"]

        agentname = "COPDAC"
        Agent = getattr(agents, agentname)
        agent = Agent(
            env,
            lrw=PARAMS["COPDAC"]["lrw"],
            lrv=PARAMS["COPDAC"]["lrv"],
            lrtheta=PARAMS["COPDAC"]["lrtheta"],
            w_init=PARAMS["COPDAC"]["w_init"],
            v_init=PARAMS["COPDAC"]["v_init"],
            theta_init=PARAMS["COPDAC"]["lrv"],
            maxlen=PARAMS["COPDAC"]["maxlen"],
            batch_size=PARAMS["COPDAC"]["batch_size"],
        )

        expname = "-".join([type(n).__name__ for n in (env, agent)])
        if kwargs["with_gan"]:
            expname += "-gan"
            agent.set_gan(kwargs["with_gan"], PARAMS["COPDAC"]["lrg"])

        if kwargs["with_reg"]:
            expname += "-reg"
            agent.set_reg(PARAMS["COPDAC"]["lrc"])

        histpath = os.path.join(copdacdir, expname + ".h5")
        if kwargs["continue"] is not None:
            epoch_start, i = agent.load(kwargs["continue"])
            logger = logging.Logger(path=histpath, max_len=100, mode="r+")
        else:
            epoch_start, i = 0, 0
            logger = logging.Logger(path=histpath, max_len=100)

        print(f"Training {agentname}...")

        epoch_end = epoch_start + kwargs["max_epoch"]
        for epoch in tqdm.trange(epoch_start, epoch_end):
            dataloader = gan.get_dataloader(
                samplefiles,
                keys=("state", "action", "reward", "next_state"),
                shuffle=True,
                batch_size=64
            )

            for data in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}"):
                agent.set_input(data)
                agent.train()

                if i % kwargs["save_interval"] == 0 or i == len(dataloader):
                    logger.record(
                        epoch=epoch,
                        i=i,
                        w=agent.w,
                        v=agent.v,
                        theta=agent.theta,
                        loss=agent.get_losses()
                    )

                i += 1

        logger.close()


@main.command()
@click.argument("path", nargs=-1, type=click.Path())
@click.option("--hist", "mode", flag_value="hist")
@click.option("--gan", "mode", flag_value="gan")
@click.option("--samples", type=click.Path(), default="data/samples")
def test(path, mode, **kwargs):
    if mode == "gan":
        samplefiles = utils.parse_file([kwargs["samples"]], ext="h5")
        trainfiles = utils.parse_file(path, ext="pth")
        agent = gan.GAN(
            lr=PARAMS["GAN"]["lr"],
            x_size=PARAMS["GAN"]["x_size"],
            u_size=PARAMS["GAN"]["u_size"],
            z_size=PARAMS["GAN"]["z_size"],
        )

        for trainfile in trainfiles:
            agent.load(trainfile)
            agent.eval()

            logger = logging.Logger(path="data/tmp.h5", max_len=500)

            dataloader = gan.get_dataloader(samplefiles, shuffle=False)
            for i, (state, action) in enumerate(tqdm.tqdm(dataloader)):
                fake_action = agent.get_action(state)
                state, action = map(torch.squeeze, (state, action))
                fake_action = fake_action.ravel()
                logger.record(
                    state=state, action=action, fake_action=fake_action)

            logger.close()

            print(f"Test data is saved in {logger.path}")


@main.command()
@click.argument("path", nargs=1, type=click.Path(exists=True))
@click.option("--out", "-o", default="data/run.h5")
@click.option("--with-plot", "-p", is_flag=True)
def run(path, **kwargs):
    logger = logging.Logger(
        log_dir=".", file_name=kwargs["out"], max_len=100)
    data = logging.load(path)
    expname = os.path.basename(path)
    envname, agentname, *_ = expname.split("-")
    env = getattr(envs, envname)(
        initial_perturb=[1, 0.0, 0, np.deg2rad(10)],
        dt=0.01, max_t=40, solver="rk4",
        ode_step_len=1
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

        files = utils.parse_file(kwargs["out"])
        canvas = []
        for file in tqdm.tqdm(files):
            canvas = figures.plot_single(file, canvas=canvas)

        figures.show()


def _run(env, agent, logger, expname, **kwargs):
    obs = env.reset()
    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        obs = next_obs

        if done:
            break

    env.close()
    return logger.path


@main.command()
@click.argument("path", nargs=1, type=click.Path())
@click.option("--sample", "mode", flag_value="sample")
@click.option("--hist", "mode", flag_value="hist")
@click.option("--gan", "mode", flag_value="gan")
@click.option("--copdac", "mode", flag_value="copdac")
def plot(path, mode, **kwargs):
    import figures

    if mode == "sample":
        files = utils.parse_file(path)
        canvas = []
        for file in tqdm.tqdm(files):
            canvas = figures.plot_single(file, canvas=canvas)
    if mode == "hist":
        figures.plot_hist(path)
    if mode == "gan":
        figures.plot_gan(path)
    if mode == "copdac":
        figures.train_plot(path)

    # if kwargs["train"]:
    #     figures.train_plot(path)
    # else:
    #     dataset = logging.load(path)
    #     figures.plot_mult(dataset)

    figures.show()


if __name__ == "__main__":
    main()
    # plot("data/tmp.h5")
