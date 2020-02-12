import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os

import fym.logging as logging

IMG_DIR = "img"
BASE_DIR = os.path.join("data")

os.makedirs(IMG_DIR, exist_ok=True)

plt.rc("font", **{
    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
})
plt.rc("text", usetex=True)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("figure", figsize=[6, 4])

data_run = logging.load("data/run.h5")
data_trained = logging.load("data/trained.h5")

canvas = []
fig, axes = plt.subplots(2, 2, sharex=True, num="states")
for ax in axes.flat:
    ax.mod = 1
axes[0, 0].set_ylabel(r"$V_T$ [m/s]")
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

fig, axes = plt.subplots(2, 2, sharex=True, num="hist")
axes[0, 0].set_ylabel(r"$\delta$")
axes[0, 1].set_ylabel(r"$w$")
axes[1, 0].set_ylabel(r"$v$")
axes[1, 1].set_ylabel(r"$\theta$")
axes[1, 0].set_xlabel("Epoch")
axes[1, 1].set_xlabel("Epoch")
canvas.append((fig, axes))


def plot_single(data_run, data_trained, name=None, **kwargs):
    time = data_run["time"]

    axes = canvas[0][1]
    for ax, x in zip(axes.flat, data_run["state"].T):
        ln, = ax.plot(time, x * ax.mod, **kwargs)
    ln.set_label(name)

    axes = canvas[1][1]
    for ax, u in zip(axes.flat, data_run["action"].T):
        ln, = ax.plot(time, u * ax.mod, **kwargs)
    ln.set_label(name)

    epoch = data_trained["epoch"]

    axes = canvas[2][1]
    axes[0, 0].plot(
        epoch,
        data_trained["delta"].reshape(-1, data_trained["delta"][0].size),
        **kwargs
    )
    axes[0, 1].plot(
        epoch,
        data_trained["w"].reshape(-1, data_trained["w"][0].size),
        **kwargs
    )
    axes[1, 0].plot(
        epoch,
        data_trained["v"].reshape(-1, data_trained["v"][0].size),
        **kwargs
    )
    ln, *_ = axes[1, 1].plot(
        epoch,
        data_trained["theta"].reshape(
            -1, np.multiply(*data_trained["theta"][0].shape)),
        **kwargs
    )
    ln.set_label(name)


plot_single(
    data_run["BaseEnv-COPDAC"],
    data_trained["BaseEnv-COPDAC"],
    name="COPDAC",
    color="r",
    linestyle="--",
)
# plot_single(
#     data_run["BaseEnv-RegCOPDAC"],
#     data_trained["BaseEnv-RegCOPDAC"],
#     name="Constrained-COPDAC",
#     color="b",
#     linestyle="-",
# )

for window in canvas:
    fig, axes = window
    fig.legend(
        *axes[-1, -1].get_legend_handles_labels(),
        bbox_to_anchor=(0.2, 0.94, 0.6, .05),
        loc='lower center',
        ncol=2,
        mode="expand",
        borderaxespad=0.
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(os.path.join(IMG_DIR, fig.canvas.get_window_title() + ".pdf"))

plt.show()
