from types import SimpleNamespace
import numpy as np


SAMPLE = SimpleNamespace(
    PERTURB=np.vstack((1, np.vstack(np.deg2rad((2, 0, 2))))),
    W_INIT=0.0,
    TIMESTEP=0.01,
    FINALTIME=20,
    LINE=SimpleNamespace(
        TMP=dict(
            color="r",
            lw=2
        ),
        STORED=dict(
            color="k",
            lw=1
        ),
    ),
    LABELS=SimpleNamespace(
        Y=[r"$V_T$ (m/s)", r"$\alpha$ (deg)", r"$Q$ (deg/s)", r"$\gamma$ (deg)",
           r"$\delta_t$ (deg)", r"$\delta_e$ (deg)", r"$\eta_1$", r"$\eta_2$"],
        X="Time (s)"
    ),
)

LOG_DIR = "log"
IMG_DIR = "img"
