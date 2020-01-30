import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import fym.logging as logging

canvas = []
fig, axes = plt.subplots(2, 2, sharex=True, num="states")
for ax in axes.flat:
    ax.mod = 1
axes[0, 0].set_ylabel(r"$v$ [m/s]")
axes[0, 1].set_ylabel(r"$\alpha$ [deg]")
axes[0, 1].mod = np.rad2deg(1)
axes[1, 0].set_ylabel(r"$q$ [deg/s]")
axes[1, 0].mod = np.rad2deg(1)
axes[1, 1].set_ylabel(r"$\gamma$ [deg]")
axes[1, 1].mod = np.rad2deg(1)
axes[1, 0].set_xlabel("time [s]")
axes[1, 1].set_xlabel("time [s]")
canvas.append((fig, axes))

fig, axes = plt.subplots(2, 2, sharex=True, num="control")
for ax in axes.flat:
    ax.mod = 1
axes[0, 0].set_ylabel(r"$\delta_t$")
axes[0, 1].set_ylabel(r"$\delta_e$ [deg]")
axes[0, 1].mod = np.rad2deg(1)
axes[1, 0].set_ylabel(r"$\eta_1$")
axes[1, 1].set_ylabel(r"$\eta_2$")
axes[1, 0].set_xlabel("time [s]")
axes[1, 1].set_xlabel("time [s]")
canvas.append((fig, axes))

fig, axes = plt.subplots(1, 1, sharex=True, squeeze=False, num="reward")
axes[0, 0].set_ylabel("reward")
axes[0, 0].set_xlabel("time [s]")
canvas.append((fig, axes))


def plot_single(data, color="k", name=None):
    time = data["time"]

    axes = canvas[0][1]
    for ax, x in zip(axes.flat, data["state"].T):
        ln, = ax.plot(time, x * ax.mod, color=color)
    ln.set_label(name)

    axes = canvas[1][1]
    for ax, u in zip(axes.flat, data["action"].T):
        ln, = ax.plot(time, u * ax.mod, color=color)
    ln.set_label(name)

    axes = canvas[2][1]
    ln, = axes[0, 0].plot(time, data["reward"], color=color)
    ln.set_label(name)

    for window in canvas:
        fig, axes = window
        axes[0, 0].legend(*axes[-1, -1].get_legend_handles_labels())
        fig.tight_layout()


def plot_mult(dataset, color_cycle=None, names=None):
    if color_cycle is None:
        color_cycle = cycler(
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

    if names is not None:
        for data, color, name in zip(dataset.values(), color_cycle(), names):
            plot_single(data, color=color["color"], name=name)

        for fig, axes in canvas:
            axes[0].legend(*axes[0].get_legend_handles_labels())
    else:
        for (name, data), color in zip(dataset.items(), color_cycle()):
            plot_single(data, color=color["color"], name=name)

    plt.show()


def train_plot(savepath):
    data = logging.load(savepath)
    epoch = data["epoch"]

    canvas = []
    fig, axes = plt.subplots(3, 1, sharex=True, squeeze=False)
    axes[0, 0].set_ylabel(r"$w$")
    axes[1, 0].set_ylabel(r"$v$")
    axes[2, 0].set_ylabel(r"$\theta$")
    axes[2, 0].set_xlabel("epoch")
    canvas.append((fig, axes))

    axes = canvas[0][1]
    axes[0, 0].plot(
        epoch,
        data["w"].reshape(-1, data["w"][0].size),
        color="k"
    )
    axes[1, 0].plot(
        epoch,
        data["v"].reshape(-1, data["v"][0].size),
        color="k"
    )
    axes[2, 0].plot(
        epoch,
        data["theta"].reshape(-1, np.multiply(*data["theta"][0].shape)),
        color="k"
    )


def show():
    plt.show()
