import click
import numpy as np
from tqdm import tqdm, trange
import os
import shutil
import glob
import time
import functools
from concurrent.futures import ProcessPoolExecutor
import copy

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
            "logging_off": False,
        },
    },
    "train": {
        "batch_size": 64,
        "max_epoch": 10,
        "print_interval": 30,
        "OPDAC": {
            "net_option": {
                "net_q": {},
                "net_pi": {},
            },
            "lrw": 1e-2,
            "lrv": 1e-2,
            "lrtheta": 1e-3,
            "w_init": 0.03,
            "v_init": 0.03,
            "theta_init": 0,
            "v_deg": 2,
        },
    },
    "run": {
        "FixedParamEnv": {
            "initial_perturb": [0, 0, 0, np.deg2rad(5)],
            "max_t": 100,
            "solver": "rk4",
            "dt": 0.01,
            "ode_step_len": 1,
            "logging_off": False,
        },
    },
    "nets": {
        "QNet": {
            "x_size": 4,
            "u_size": 4,
            "lr": 1e-4,
            "gamma": 0.9999,
        },
        "PolyNet": {
            "x_size": 4,
            "u_size": 4,
            "lr": 1e-5,
            "degree": 2,
        },

    },
    "GAN": {
        "x_size": 4,
        "u_size": 4,
        "z_size": 100,
        "lr": 2e-4,
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
        list(tqdm(
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
    param = {**PARAMS["sample"]["OuNoise"], "mean": np.zeros((4, 4))}
    ou_noise = common.OuNoise(**param)
    behavior.add_noise(ou_noise)
    env.set_inner_ctrl(behavior)
    env.set_logger_callback()
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
@click.option("--agent", default="OPDAC")
@click.option("--with", type=click.Choice(["qzero"]))
@click.option("--gan-type", default="g", type=click.Choice(["d", "g"]))
@click.option("--gan-lr", default=PARAMS["GAN"]["lr"])
@click.option("--use-cuda", is_flag=True)
@click.option("--gan-dir", default="data/gan")
@click.option("--copdac-dir", default="data/copdac")
@click.option("--continue", "-c", nargs=1, type=click.Path(exists=True))
@click.option("--max-epoch", "-n", default=100)
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
        for epoch in trange(epoch_start,
                            epoch_start + 1 + kwargs["max_epoch"]):
            dataloader = gan.get_dataloader(
                samplefiles, shuffle=True, batch_size=kwargs["batch_size"])

            loss_d = loss_g = 0
            for i, data in enumerate(tqdm(dataloader)):
                agent.set_input(data)
                agent.train()
                loss_d += agent.loss_d.mean().detach().numpy()
                loss_g += agent.loss_g.mean().detach().numpy()

            logger.record(epoch=epoch, loss_d=loss_d, loss_g=loss_g)

            if (epoch % kwargs["save_interval"] == 0
                    or epoch == epoch_start + 1 + kwargs["max_epoch"]):
                savepath = os.path.join(gandir, f"trained-{epoch:05d}.pth")
                agent.save(epoch, savepath)
                tqdm.write(f"Weights are saved in {savepath}.")

        print(f"Elapsed time: {time.time() - t0:5.2f} sec")

    if mode == "copdac" or mode == "all":
        np.random.seed(kwargs["seed"])

        envparams = PARAMS["sample"]["FixedParamEnv"]
        envparams["logging_off"] = True
        env = envs.FixedParamEnv(**envparams)

        agentclass = kwargs["agent"]
        agentparams = PARAMS["train"][agentclass]
        agentparams["net_option"]["net_q"] = {
            "QNet": PARAMS["nets"]["QNet"]}
        agentparams["net_option"]["net_pi"] = {
            "PolyNet": PARAMS["nets"]["PolyNet"]}
        agent = getattr(agents, agentclass)(env, **agentparams)

        if "qzero" in kwargs["with"]:
            agent.set_addon("qzero")

        expname = "-".join((env.name, agent.get_name()))
        expdir = os.path.join(kwargs["copdac_dir"], expname)
        histpath = os.path.join(expdir, "hist.h5")

        if kwargs["continue"] is not None:
            epoch_init, global_step, mode = agent.load(kwargs["continue"])
        else:
            if os.path.exists(expdir):
                if input(f"Delete \"{expdir}\"? [Y/n]: ") in ["", "Y", "y"]:
                    shutil.rmtree(expdir)

            epoch_init, global_step, mode = 0, 0, "w"

        logger = logging.Logger(path=histpath, max_len=1, mode=mode)
        logger.set_info(
            env=env.__class__.__name__,
            agent=agent.__class__.__name__,
            expname=expname,
            envparams=envparams,
            agentparams=agentparams,
            PARAMS=PARAMS,
            click=kwargs,
        )

        print(f"Training {expname} ...")

        def trimming(x, u, r, nx):
            return x - env.trim_x, u - env.trim_u, r, nx - env.trim_x

        dataloader = common.get_dataloader(
            samplefiles,
            keys=("state", "action", "reward", "next_state"),
            shuffle=True,
            batch_size=PARAMS["train"]["batch_size"],
            transform=trimming
        )

        max_global = kwargs["max_epoch"] * len(dataloader)
        logging_interval = int(1e-2 * max_global) or 1
        save_interval = int(1e-1 * max_global) or 1

        epoch_final = epoch_init + kwargs["max_epoch"]
        t0 = time.time()
        for epoch in range(epoch_init, epoch_final):
            desc = f"Epoch {epoch:2d}/{epoch_final - 1:2d}"
            for n, data in enumerate(tqdm(dataloader, desc=desc, leave=False)):
                agent.set_input(data)
                agent.update()

                if global_step % PARAMS["train"]["print_interval"] == 0:
                    msg = "\t".join([
                        f"[Step_{global_step:07d}]",
                        agent.get_msg()
                    ])
                    tqdm.write(msg)

                if (global_step % logging_interval == 0
                        or global_step == len(dataloader)):
                    logger.record(
                        epoch=epoch,
                        global_step=global_step,
                        state_dict=copy.deepcopy(agent.state_dict()),
                        loss=agent.info["loss"]
                    )
                    tqdm.write("Recorded")

                if (global_step % save_interval == 0
                        or global_step == len(dataloader)):
                    savepath = os.path.join(expdir,
                                            f"trained-{global_step:07d}.pth")
                    agent.save(epoch, global_step, savepath)

                global_step += 1

        logger.close()

        print(f"Elapsed time: {time.time() - t0:5.2f} sec")
        print(f"Exp. saved in \"{expdir}\"")


@main.command()
@click.argument("histpath", nargs=1, type=click.Path(exists=True))
@click.option("--out", "-o", default="data/run.h5")
@click.option("--with-plot", "-p", is_flag=True)
def run(histpath, **kwargs):
    data, info = logging.load(histpath, with_info=True)
    expdir = os.path.dirname(histpath)
    weightpath = sorted(glob.glob(os.path.join(expdir, "*.pth")))[-1]
    weight = torch.load(weightpath)

    envparams = PARAMS["run"][info["env"]]
    envparams["logging_path"] = kwargs["out"]
    env = getattr(envs, info["env"])(**envparams)

    agent = getattr(agents, info["agent"])(env, **info["agentparams"])
    agent.load_weights(weight)
    agent.eval()

    env.set_logger_callback()
    env.set_inner_ctrl(agent)

    print(f"Runnning {info['expname']} ...")

    env.reset()
    while True:
        env.render()
        _, _, done, info = env.step()

        if done:
            break

    env.close()

    if kwargs["with_plot"]:
        import figures

        files = utils.parse_file(kwargs["out"])
        canvas = []
        for file in tqdm(files):
            canvas = figures.plot_single(file, canvas=canvas)

        canvas.append(figures.train_plot(histpath))

        figures.show()


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
            for i, (state, action) in enumerate(tqdm(dataloader)):
                fake_action = agent.get_action(state)
                state, action = map(torch.squeeze, (state, action))
                fake_action = fake_action.ravel()
                logger.record(
                    state=state, action=action, fake_action=fake_action)

            logger.close()

            print(f"Test data is saved in {logger.path}")


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
        for file in tqdm(files):
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

    # for x, u, nu in tqdm(dataloader):
    #     plt.plot(x, (u - nu).numpy() / env.trim_u,
    #              ".", ms=2, mew=0, mfc=(0, 0, 0, 1))

    # plt.show()
