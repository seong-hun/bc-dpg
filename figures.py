import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging


def plot(savepath):
    data = logging.load(savepath)

    canvas = []
    fig, axes = plt.subplots(2, 2, sharex=True)
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

    fig, axes = plt.subplots(2, 2, sharex=True)
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

    time = data["time"]

    axes = canvas[0][1]
    for ax, x in zip(axes.flat, data["state"].T):
        ax.plot(time, x * ax.mod, color="k")
    # axes[0, 0].lines[0].set_label("True")
    # axes[0, 0].legend(*axes[0].get_legend_handles_labels())

    axes = canvas[1][1]
    for ax, u in zip(axes.flat, data["control"].T):
        ax.plot(time, u * ax.mod, color="k")
    # axes[0, 0].lines[0].set_label("True")
    # axes[0, 0].legend(*axes[0].get_legend_handles_labels())

    for fa in canvas:
        fa[0].tight_layout()

    plt.show()
