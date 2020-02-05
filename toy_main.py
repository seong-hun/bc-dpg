import click
import numpy as np
import tqdm
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import shutil

import torch
import torch.multiprocessing as mp

import fym.logging as logging

import envs
import agents
import gan

import matplotlib.pyplot as plt


def behavior_policy(x):
    if np.random.rand() > 0.5:
        return 2 * x + 0.4 * np.random.randn()
    else:
        return 0 + 0.3 * np.random.randn()


@click.group()
def main():
    pass


@main.command()
@click.option("--base-dir", default="data/toy")
@click.option("--max-epoch", "-n", "max_epoch", default=1000)
@click.option("--batch-size", "-b", default=64)
def train(**kwargs):
    np.random.seed(0)

    print("Data generation ...")
    sampledir = os.path.join(kwargs["base_dir"], "samples")
    logger = logging.Logger(
        log_dir=sampledir, file_name="sample.h5", max_len=100)

    for _ in tqdm.trange(10000):
        x = 2 * np.random.randn()
        u = behavior_policy(x)
        logger.record(state=[x], action=[u])

    logger.close()

    sample_files = sorted(glob.glob(os.path.join(sampledir, "*.h5")))
    if isinstance(sample_files, str):
        sample_files = (sample_files, )

    print("Train GAN ...")
    gandir = os.path.join(kwargs["base_dir"], "gan")
    try:
        shutil.rmtree(gandir)
    except FileNotFoundError:
        os.makedirs(gandir, exist_ok=True)

    save_interval = int(kwargs["max_epoch"] / 10)

    sample_files = sorted(glob.glob(os.path.join(sampledir, "*.h5")))
    agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)
    prog = partial(
        _gan_prog, agent=agent, files=sample_files,
        batch_size=kwargs["batch_size"],
    )

    t0 = time.time()
    for epoch in tqdm.trange(kwargs["max_epoch"]):
        prog(epoch)

        if epoch % save_interval == 0:
            agent.save(epoch, os.path.join(gandir, f"trained_{epoch:05d}.pth"))

#     agent.share_memory()
#     with mp.Pool(kwargs["max_workers"] or None) as p:
#         list(tqdm.tqdm(
#             p.map(prog, range(kwargs["max_epoch"])),
#             total=kwargs["max_epoch"],
#         ))

    print(f"Elapsed time: {time.time() - t0:5.2f} sec")


def _gan_prog(epoch, agent, files, shuffle=True, batch_size=32):
    dataloader = gan.get_dataloader(
        files, shuffle=shuffle, batch_size=batch_size)

    for i, data in enumerate(dataloader):
        agent.set_input(data)
        agent.train()


@main.command()
@click.option("--base-dir", default="data/toy")
def plot(**kwargs):
    sampledir = os.path.join(kwargs["base_dir"], "samples")
    gandir = os.path.join(kwargs["base_dir"], "gan")

    sample_files = sorted(glob.glob(os.path.join(sampledir, "*.h5")))

    agent = gan.GAN(lr=1e-3, x_size=1, u_size=1, z_size=10)
    best_weights = sorted(glob.glob(os.path.join(gandir, "*.pth")))[-1]
    agent.load(best_weights)
    print(f"Using {best_weights} ...")

    data_all = [logging.load(name) for name in sample_files]
    x, u = [
        np.vstack([data[k] for data in data_all])
        for k in ("state", "action")
    ]
    plt.scatter(x, u)
    plt.scatter(x, agent.get_action(x))

    plt.show()


if __name__ == "__main__":
    main()
