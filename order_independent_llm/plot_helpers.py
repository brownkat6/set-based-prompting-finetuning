import os
import os.path

import matplotlib.pyplot as plt

rc_params = {
    "figure.figsize": [9, 5],
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.top": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "ytick.right": False,
    "legend.fancybox": False,
    "legend.shadow": False,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
    "legend.markerscale": 2,
    "legend.framealpha": 0.5,
    "errorbar.capsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    #'font.sans-serif': ['Helvetica'], #Requires Helvetica installed
}


def nice_defaults() -> dict:
    plt.rcParams.update(rc_params)
    return plt.rcParams


def multi_savefig(
    save_name, dir_name="images", dpi=300, save_types=("pdf", "png", "svg")
):
    os.makedirs(dir_name, exist_ok=True)
    for sType in save_types:
        dName = os.path.join(dir_name, sType)
        os.makedirs(dName, exist_ok=True)

        fname = f"{save_name}.{sType}"

        plt.savefig(
            os.path.join(dName, fname),
            format=sType,
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
        )
