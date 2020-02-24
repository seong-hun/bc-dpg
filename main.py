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
import common

PARAMS = {
    "sample": {
        "OuNoise": {
            "mean": 0,
            "sigma": 0.01,
            "dt": 1,
            "max_t": 100,
            "decay": 30
        },
        "FixedParamEnv": {
            "initial_perturb": [0, 0, 0, 0.1],
            "max_t": 100,
            "solver": "odeint",
            "dt": 40,
            "ode_step_len": 1000,
            "logging_off": False
        },
    },
    "BaseEnv": {
        "phi_deg": 1,
        "dt": 0.01,
        "max_t": 20,
        "solver": "rk4",
        "ode_step_len": 1,
    },
    "GAN": {
        "x_size": 4,
        "u_size": 4,
        "z_size": 100,
        "lr": 2e-4,
    },
    "COPDAC": {
        "lrw": 1e-2,
        "lrv": 1e-2,
        "lrtheta": 1e-3,
        "w_init": 0.03,
        "v_init": 0.03,
        "theta_init": 0,
        "v_deg": 2,
        "maxlen": 100,
        "batch_size": 64,
    },
    "addons": {
        "reg": {
            "lrc": 2e-3,
        },
        "GAN": {
            "lrg": 1e-1,
        },
        "const": {
        },
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

    env = envs.FixedParamEnv(
        **PARAMS["sample"]["FixedParamEnv"],
        logging_path=os.path.join(log_dir, f"{i:03d}.h5")
    )
    behavior = agents.Linear(xdim=4, udim=4)
    behavior.set_param(0)
    ou_noise = common.OuNoise(**PARAMS["sample"]["OuNoise"])
    behavior.add_noise(ou_noise)
    env.set_inner_ctrl(behavior)
    env.reset("random")

    while True:
        _, _, done, _ = env.step()

        if done:
            break

    env.logger.set_info(
        initial_state=env.system.initial_state,
        **PARAMS["sample"],
    )

    env.close()


@main.command()
@click.argument("sample", nargs=-1, type=click.Path())
@click.option("--all", "mode", flag_value="all", default=True)
@click.option("--gan", "mode", flag_value="gan")
@click.option("--copdac", "mode", flag_value="copdac")
@click.option("--gan-type", default="g", type=click.Choice(["d", "g"]))
@click.option("--gan-lr", default=PARAMS["GAN"]["lr"])
@click.option("--use-cuda", is_flag=True)
@click.option("--gan-dir", default="data/gan")
@click.option("--copdac-dir", default="data/copdac")
@click.option("--continue", "-c", nargs=1, type=click.Path(exists=True))
@click.option("--max-epoch", "-n", default=100)
@click.option("--save-interval", "-s", default=10)
@click.option("--batch-size", "-b", default=64)
@click.option("--with-reg", is_flag=True)
@click.option("--with-const", is_flag=True)
@click.option("--with-gan", nargs=1, type=click.Path(exists=True))
@click.option("--out", "-o", "savepath", default="data/trained.h5")
@click.option("--seed", default=0)
def train(sample, mode, **kwargs):
    samplefiles = utils.parse_file(sample, ext="h5")

    if mode == "gan" or mode == "all":
        torch.manual_seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

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
        np.random.seed(kwargs["seed"])

        env = envs.FixedParamEnv(**PARAMS["sample"]["FixedParamEnv"])
        agent = agents.COPDAC(env, **PARAMS["COPDAC"])

        # Add-ons
        if kwargs["with_gan"]:
            agent.set_gan(
                kwargs["with_gan"],
                **PARAMS["addons"]["GAN"],
                gan_type=kwargs["gan_type"],
            )

        if kwargs["with_reg"]:
            agent.set_reg(**PARAMS["addons"]["reg"])

        if kwargs["with_const"]:
            agent.set_const(**PARAMS["addons"]["const"])

        copdacdir = kwargs["copdac_dir"]
        expname = "-".join((env.name, agent.get_name()))
        histpath = os.path.join(copdacdir, expname + ".h5")

        if kwargs["continue"] is not None:
            epoch_start, i = agent.load(kwargs["continue"])
            logger = logging.Logger(path=histpath, max_len=100, mode="r+")
        else:
            epoch_start, i = 0, 0
            logger = logging.Logger(path=histpath, max_len=100)

        logger.set_info(
            env=env.__class__.__name__,
            agent=agent.__class__.__name__,
            PARAMS=PARAMS,
            click=kwargs,
        )

        print(f"Training {expname}...")

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
    data, info = logging.load(path, with_info=True)

    env = getattr(envs, info["env"])(
        initial_perturb=[0, 0.0, 0, np.deg2rad(10)],
        **info["PARAMS"]["BaseEnv"])
    agent = getattr(agents, info["agent"])(env, **info["PARAMS"]["COPDAC"])
    agent.load_weights(data)

    expname = os.path.splitext(os.path.basename(path))[0]

    print(f"Runnning {expname} ...")

    logger = logging.Logger(path=kwargs["out"], max_len=100)

    _run(env, agent, logger, expname, **kwargs)

    logger.close()

    if kwargs["with_plot"]:
        import figures

        files = utils.parse_file(kwargs["out"])
        canvas = []
        for file in tqdm.tqdm(files):
            canvas = figures.plot_single(file, canvas=canvas)

        canvas.append(figures.train_plot(path))

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
    # env = envs.BaseEnv(**PARAMS["BaseEnv"])
    # agent = agents.COPDAC(env, **PARAMS["COPDAC"])

    # samplefiles = utils.parse_file("data/samples", ext="h5")
    # dataloader = gan.get_dataloader(
    #     samplefiles, keys=("state", "action", "next_action"),
    #     shuffle=True, batch_size=64)

    # import matplotlib.pyplot as plt

    # for x, u, nu in tqdm.tqdm(dataloader):
    #     plt.plot(x, (u - nu).numpy() / env.trim_u,
    #              ".", ms=2, mew=0, mfc=(0, 0, 0, 1))

    # plt.show()
