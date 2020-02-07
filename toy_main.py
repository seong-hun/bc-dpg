from concurrent.futures import ProcessPoolExecutor
import click
import functools
import glob
import numpy as np
import os
import shutil
import time
import tqdm

import torch
import torch.multiprocessing as mp

import fym.logging as logging

import agents
import envs
import gan

import matplotlib.pyplot as plt


def common_params(func):
    @click.option("--base-dir", default="data/toy")
    @click.option("--sample-dir", default="data/toy/samples")
    @click.option("--gan-dir", default="data/toy/gans")
    @click.option("--test-dir", default="data/toy/tests")
    @click.option("--img-dir", default="img/toy")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@click.group(invoke_without_command=True)
def main():
    ctx = click.get_current_context()
    ctx.ensure_object(dict)

    if ctx.invoked_subcommand is None:
        ctx.invoke(sample, all=True)
        ctx.invoke(train, samples=glob.glob("data/toy/samples/*"))
        ctx.invoke(test, weightfiles=glob.glob("data/toy/gans/**/*.pth"))
        # ctx.invoke(plot)


@main.command()
@common_params
@click.option("--all", "-a", is_flag=True)
@click.option("--mode", "-m", type=click.Choice(["even", "sparse", "det"]),
              multiple=True)
def sample(**kwargs):
    np.random.seed(0)
    exps = kwargs["mode"]
    if kwargs["all"]:
        exps = ("even", "sparse", "det")

    for exp in exps:
        samplepath = os.path.join(kwargs["sample_dir"], exp + ".h5")

        print(f"Sample for {exp} ...")
        get_data = DataGen(exp)

        logger = logging.Logger(path=samplepath, max_len=500)

        t0 = time.time()
        for _ in tqdm.trange(10000):
            x, u, mask = get_data()
            logger.record(state=[x], action=[u], mask=[mask])

        logger.close()
        print(f"Saved in {samplepath}.")
        print(f"Elapsed time: {time.time() - t0:5.2f} seconds.")


class DataGen():
    x_bound = (-1, 1)
    u_bound = (-2, 2)

    def __init__(self, mode):
        self.get_behavior = self._get_behavior(noise=0.1)
        self.get_mask = self._even_mask

        if mode == "sparse":
            self.get_mask = self._sparse_mask
        elif mode == "det":
            self.get_behavior = self._get_behavior(noise=0)

    def __call__(self):
        x = np.clip(self.get_x(), *self.x_bound)
        u = np.clip(self.get_behavior(x), *self.u_bound)
        mask = self.get_mask(x)
        return x, u, mask

    def get_x(self):
        return np.random.uniform(*self.x_bound)

    def _even_mask(self, x):
        return 0

    def _sparse_mask(self, x):
        if -0.7 < x < -0.25 or 0.3 < x < 0.6 or x > 0.8:
            return 1  # not used for training
        else:
            return 0

    def _get_behavior(self, noise):
        def func(x, noise=noise):
            if np.random.rand() > 0.5:
                return 2 * x + noise * np.random.randn()
            else:
                return 0 + noise * np.random.randn()
        return func


@main.command()
@common_params
@click.argument("samples", nargs=-1, type=click.Path(exists=True))
@click.option("--max-epoch", "-n", "max_epoch", default=500)
@click.option("--batch-size", "-b", default=64)
@click.option("--max-workers", "-m", default=0)
def train(samples, **kwargs):
    for sample in samples:
        basedir = os.path.relpath(sample, kwargs["sample_dir"])

        if os.path.isdir(sample):
            samplefiles = sorted(glob.glob(os.path.join(sample, "*.h5")))
        elif os.path.isfile(sample):
            samplefiles = sample
            basedir = os.path.splitext(basedir)[0]
        else:
            raise ValueError("unknown sample type.")

        gandir = os.path.join(kwargs["gan_dir"], basedir)

        if os.path.exists(gandir):
            shutil.rmtree(gandir)
        os.makedirs(gandir, exist_ok=True)

        print(f"Train GAN for sample ({sample}) ...")

        save_interval = int(kwargs["max_epoch"] / 10)

        agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)
        prog = functools.partial(
            _gan_prog, agent=agent, files=samplefiles,
            batch_size=kwargs["batch_size"],
        )

        logger = logging.Logger(
            path=os.path.join(gandir, "train_history.h5"), max_len=500)

        t0 = time.time()
        for epoch in tqdm.trange(kwargs["max_epoch"]):
            loss_d, loss_g = prog(epoch)

            logger.record(epoch=epoch, loss_d=loss_d, loss_g=loss_g)

            if epoch % save_interval == 0:
                agent.save(
                    epoch,
                    os.path.join(gandir, f"trained_{epoch:05d}.pth")
                )

        print(f"Elapsed time: {time.time() - t0:5.2f} sec")
        torch.save(
            samplefiles,
            os.path.join(gandir, "sample_path.h5"),
        )


def _gan_prog(epoch, agent, files, shuffle=True, batch_size=32):
    if isinstance(files, str):
        files = [files]

    dataloader = gan.get_dataloader(
        files, shuffle=shuffle, batch_size=batch_size)

    loss_d = 0
    loss_g = 0
    for i, (x, u, mask) in enumerate(dataloader):
        x = x[~mask.bool().squeeze()]
        u = u[~mask.bool().squeeze()]
        agent.set_input((x, u))
        agent.train()
        loss_d += agent.loss_d.mean().detach().numpy()
        loss_g += agent.loss_d.mean().detach().numpy()

    return loss_d / i, loss_g / i


@main.command()
@common_params
@click.argument("weightfiles", nargs=-1, type=click.Path(exists=True))
@click.option("--plot", "-p", is_flag=True)
def test(**kwargs):
    agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)

    for weightfile in tqdm.tqdm(kwargs["weightfiles"]):
        tqdm.tqdm.write(f"Using {weightfile} ...")

        agent.load(weightfile)
        gandir = os.path.dirname(weightfile)
        testpath = os.path.join(
            kwargs["test_dir"],
            os.path.splitext(
                os.path.relpath(weightfile, kwargs["gan_dir"]))[0]
            + ".h5"
        )
        samplefiles = torch.load(os.path.join(gandir, "sample_path.h5"))

        if isinstance(samplefiles, str):
            samplefiles = [samplefiles]

        data_all = [logging.load(name) for name in samplefiles]
        real_x, real_u, mask = [
            np.vstack([data[k] for data in data_all])
            for k in ("state", "action", "mask")
        ]
        data_gen = DataGen(mode="even")
        fake_x = np.zeros_like(real_x)
        fake_u = np.zeros_like(real_u)
        for i in range(fake_x.size):
            fake_x[i] = data_gen()[0]
        fake_u = agent.get_action(fake_x)

        logging.save(testpath, dict(
            real_x=real_x, real_u=real_u, mask=mask,
            fake_x=fake_x, fake_u=fake_u))
        tqdm.tqdm.write(f"Test data is saved in {testpath}.")


@main.command()
@common_params
@click.argument("testfiles", nargs=-1, type=click.Path(exists=True))
def plot(**kwargs):
    os.makedirs(kwargs["img_dir"], exist_ok=True)

    plt.rc("font", **{
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
    })
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)
    plt.rc("figure", figsize=[6, 4])

    for testfile in kwargs["testfiles"]:
        _plot(testfile)

    plt.show()


def _plot(testfile):
    data = logging.load(testfile)
    real_x, real_u, mask, fake_x, fake_u = (
        data[k].ravel()
        for k in ("real_x", "real_u", "mask", "fake_x", "fake_u"))

    xmin, xmax = real_x.min(), real_x.max()
    umin, umax = real_u.min(), real_u.max()
    canvas = []

    fig, axes = plt.subplots(1, 2, sharey=True, squeeze=False, num="real")
    axes[0, 0].set_ylabel(r"$u$")

    axes[0, 0].set_xlabel(r"$x$")
    axes[0, 1].set_xlabel(r"$x$")

    axes[0, 0].set_xlim([xmin, xmax])
    axes[0, 1].set_xlim([xmin, xmax])
    axes[0, 0].set_ylim([umin, umax])

    axes[0, 0].set_title("Real")
    axes[0, 1].set_title("Fake")

    canvas.append((fig, axes))

    fig, axes = canvas[0]
    ax = axes[0, 0]
    # ax.imshow(np.rot90(Z), cmap=plt.cm.get_cmap("hot"),
    #           extent=[xmin, xmax, umin, umax], aspect="auto")
    mask = mask.astype(bool)
    ax.plot(real_x[~mask], real_u[~mask], '.', markersize=2,
            mew=0, mfc=(0, 0, 0, 1))
    ax.plot(real_x[mask], real_u[mask], '.', markersize=2,
            mew=0, mfc=(1, 0, 0, 0.1))

    ax = axes[0, 1]
    ax.plot(fake_x, fake_u, '.', markersize=2,
            mew=0, mfc=(0, 0, 0, 1))

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
