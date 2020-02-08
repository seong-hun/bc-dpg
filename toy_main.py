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


class DataStructure:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.sample_dir = os.path.join(base_dir, "samples")
        self.gan_dir = os.path.join(base_dir, "gans")
        self.test_dir = os.path.join(base_dir, "tests")
        self.img_dir = os.path.join(base_dir, "imgs")


@click.group(invoke_without_command=True)
@click.option("--base-dir", default="data/toy")
@click.pass_context
def main(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj = DataStructure(kwargs["base_dir"])

    if ctx.invoked_subcommand is None:
        ctx.invoke(sample, all=True)
        ctx.invoke(train, samples=glob.glob("data/toy/samples/*"))
        ctx.invoke(test, weightfiles=glob.glob("data/toy/gans/**/*.pth"))
        # ctx.invoke(plot)


@main.command()
@click.option("--all", "-a", is_flag=True)
@click.option("--mode", "-m", type=click.Choice(["even", "sparse", "det"]),
              multiple=True)
@click.option("--noise", "-n", default=0.15)
@click.pass_obj
def sample(obj, **kwargs):
    np.random.seed(0)
    exps = kwargs["mode"]
    if kwargs["all"]:
        exps = ("even", "sparse", "det")

    for exp in exps:
        samplepath = os.path.join(obj.sample_dir, exp + ".h5")

        print(f"Sample for {exp} ...")
        get_data = DataGen(exp, noise=kwargs["noise"])

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

    def __init__(self, mode, noise=0):
        self.get_behavior = self._get_behavior(noise=noise)
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
        return True

    def _sparse_mask(self, x):
        if -0.8 < x < -0.5 or 0.3 < x < 0.6 or x > 0.8:
            return False  # not used for training
        else:
            return True

    def _get_behavior(self, noise):
        def func(x, noise=noise):
            if np.random.rand() > 0.5:
                return 2 * x + noise * np.random.randn()
            else:
                return 0 + noise * np.random.randn()
        return func


@main.command()
@click.argument("samples", nargs=-1, type=click.Path(exists=True))
@click.option("--max-epoch", "-n", "max_epoch", default=100)
@click.option("--save-interval", "-s", default=10)
@click.option("--batch-size", "-b", default=64)
@click.option("--max-workers", "-m", default=0)
@click.pass_obj
def train(obj, samples, **kwargs):
    for sample in samples:
        basedir = os.path.relpath(sample, obj.sample_dir)

        if os.path.isdir(sample):
            samplefiles = sorted(glob.glob(os.path.join(sample, "*.h5")))
        elif os.path.isfile(sample):
            samplefiles = sample
            basedir = os.path.splitext(basedir)[0]
        else:
            raise ValueError("unknown sample type.")

        gandir = os.path.join(obj.gan_dir, basedir)

        if os.path.exists(gandir):
            shutil.rmtree(gandir)
        os.makedirs(gandir, exist_ok=True)

        print(f"Train GAN for sample ({sample}) ...")

        save_interval = int(kwargs["save_interval"])

        agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)
        prog = functools.partial(
            _gan_prog, agent=agent, files=samplefiles,
            batch_size=kwargs["batch_size"],
        )

        logger = logging.Logger(
            path=os.path.join(gandir, "train_history.h5"), max_len=500)

        t0 = time.time()
        for epoch in tqdm.trange(1 + kwargs["max_epoch"]):
            loss_d, loss_g = prog(epoch)

            logger.record(epoch=epoch, loss_d=loss_d, loss_g=loss_g)

            if epoch % save_interval == 0 or epoch == 1 + kwargs["max_epoch"]:
                savepath = os.path.join(gandir, f"trained_{epoch:05d}.pth")
                agent.save(epoch, savepath)
                tqdm.tqdm.write(f"Weights are saved in {savepath}.")

        logger.close()

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
        x = x[mask.bool().squeeze()]
        u = u[mask.bool().squeeze()]
        agent.set_input((x, u))
        agent.train()
        loss_d += agent.loss_d.mean().detach().numpy()
        loss_g += agent.loss_d.mean().detach().numpy()

    return loss_d / i, loss_g / i


@main.command()
@common_params
@click.argument("weightfiles", nargs=-1, type=click.Path(exists=True))
@click.option("--plot", "-p", is_flag=True)
@click.pass_obj
def test(obj, **kwargs):
    agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)

    for weightfile in tqdm.tqdm(sorted(kwargs["weightfiles"])):
        tqdm.tqdm.write(f"Using {weightfile} ...")

        agent.load(weightfile)
        # agent.eval()

        gandir = os.path.dirname(weightfile)
        testpath = os.path.join(
            obj.test_dir,
            os.path.splitext(
                os.path.relpath(weightfile, obj.gan_dir))[0]
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

        _plot(obj, testpath)
        plt.show()


@main.command()
@common_params
@click.option("--sample", "mode", flag_value="sample", default=True)
@click.argument("testfiles", nargs=-1, type=click.Path(exists=True))
@click.pass_obj
def plot(obj, **kwargs):
    os.makedirs(obj.img_dir, exist_ok=True)

    plt.rc("font", **{
        "family": "sans-serif",
        "sans-serif": ["Helvetica"],
    })
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)
    plt.rc("figure", figsize=[6, 4])

    if kwargs["mode"] == "sample":
        for testfile in kwargs["testfiles"]:
            print(testfile)
            _plot_sample(testfile)

    plt.show()


def _plot_sample(testfile):
    data = logging.load(testfile)

    real_x, real_u, mask = (
        data[k].ravel()
        for k in ("state", "action", "mask"))

    xmin, xmax = real_x.min(), real_x.max()
    umin, umax = real_u.min(), real_u.max()
    canvas = []

    fig, axes = plt.subplots(1, 1, squeeze=False, num=testfile)
    axes[0, 0].set_ylabel(r"$u$")

    axes[0, 0].set_xlabel(r"$x$")

    axes[0, 0].set_xlim([xmin, xmax])
    axes[0, 0].set_ylim([umin, umax])

    canvas.append((fig, axes))

    fig, axes = canvas[0]
    ax = axes[0, 0]
    mask = mask.astype(bool)
    ax.plot(real_x[mask], real_u[mask], '.', markersize=2,
            mew=0, mfc=(0, 0, 0, 1))
    ax.plot(real_x[~mask], real_u[~mask], '.', markersize=2,
            mew=0, mfc=(1, 0, 0, 0.1))

    fig.tight_layout()
    return fig


def _plot(obj, testfile):
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
    mask = mask.astype(bool)
    ax.plot(real_x[mask], real_u[mask], '.', markersize=2,
            mew=0, mfc=(0, 0, 0, 1))
    ax.plot(real_x[~mask], real_u[~mask], '.', markersize=2,
            mew=0, mfc=(1, 0, 0, 0.1))

    ax = axes[0, 1]
    ax.plot(fake_x, fake_u, '.', markersize=2,
            mew=0, mfc=(0, 0, 0, 1))
    fig.tight_layout()


def _plot_hist(obj, testfile):
    histfile = os.path.join(
        obj.gan_dir,
        os.path.relpath(os.path.dirname(testfile), obj.test_dir),
        "train_history.h5"
    )

    histdata = logging.load(histfile)

    canvas = []

    fig, axes = plt.subplots(1, 2, sharey=True, squeeze=False, num="loss")
    axes[0, 0].set_ylabel(r"Loss")

    axes[0, 0].set_xlabel(r"Epoch")
    axes[0, 1].set_xlabel(r"Epoch")

    axes[0, 0].set_title("Generator")
    axes[0, 1].set_title("Discrimator")

    canvas.append((fig, axes))

    fig, axes = canvas[0]
    ax = axes[0, 0]
    ax.plot(histdata["epoch"], histdata["loss_g"])
    ax = axes[0, 1]
    ax.plot(histdata["epoch"], histdata["loss_d"])
    fig.tight_layout()


if __name__ == "__main__":
    main()
