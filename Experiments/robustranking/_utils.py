import itertools
from abc import ABC
import copy
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from robustranking.comparison import BootstrapComparison


# Custom aggregation methods
class PAR(ABC):
    """
    Custom aggregation function for 1-d arrays
    """
    def __init__(self, k=10, cutoff=60):
        """

        Args:
            k: the penalty factor
            cutoff: the cutoff time to be considered as timeout
        """
        self.k = k
        self.cutoff = cutoff

    def __call__(self, array: [np.ndarray | pd.Series]):
        """

        Args:
            array: numpy array
        Returns:
            PARk
        """
        array = copy.copy(array)
        array[array >= self.cutoff] = self.k * self.cutoff
        return np.mean(array)


# Visualisations
def plot_distribution(comparison: BootstrapComparison, algorithm: str, ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(9, 6))

    distribution = comparison.get_bootstrap_distribution(algorithm)
    bins = 20
    ax.hist(distribution,
            bins=20,
            density=True,
            label=f"histogram (bins={bins})",
            linestyle="-",
            color=(35/255, 134/255, 247/255),
            edgecolor=(23/255, 90/255, 166/255),
            alpha=0.5)

    histogram, edges = np.histogram(distribution, bins, density=True)
    kde = gaussian_kde(distribution)
    x = np.linspace(edges[0], edges[-1], 2048)
    ax.plot(x, kde(x), label="kernel density estimation")

    df = comparison.get_confidence_intervals()
    bounds = df.loc[algorithm, :]
    red = (237/255, 59/255, 43/255)
    ax.axvline(bounds["median"],
               color=red,
               linestyle="-",
               label="median ({:.0f})".format(bounds["median"]))
    ax.axvline(bounds["lb"],
               color=red,
               linestyle="--",
               label="{} CI bounds ({:.0f}, {:.0f})".format(1-comparison.alpha, bounds["lb"], bounds["ub"]))
    ax.axvline(bounds["ub"],
               color=red,
               linestyle="--")

    ax.set_title(algorithm)
    ax.set_xlabel("Performance")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.0))

    if show:
        plt.tight_layout()
        plt.show()

def plot_distributions_comparison(comparison: BootstrapComparison,
                                  algorithms: list,
                                  show_p_values: bool = True,
                                  ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(9, 6))

    distributions = []
    for algorithm in algorithms:
        distributions.append(comparison.get_bootstrap_distribution(algorithm))

    distributions = np.stack(distributions)

    df = comparison.get_confidence_intervals()
    for i, algorithm in enumerate(algorithms):
        kde = gaussian_kde(distributions[i, :])
        x = np.linspace(np.min(distributions), np.max(distributions), 2048)
        ax.fill_between(x, kde(x), label=algorithm, alpha=0.66, edgecolor="black")

        #CI shadow
        x = np.linspace(df.loc[algorithm, "lb"], df.loc[algorithm, "ub"], 2048)
        ax.fill_between(x, kde(x), color="black", alpha=0.1)



    if show_p_values:
        for a1, a2 in itertools.combinations(algorithms, r=2):
            print("{} vs {}: {}".format(a1, a2, comparison.statistical_test(a1, a2)))

    ax.set_xlabel("Performance")
    ax.set_ylabel("Density")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()


def plot_ci_list(comparison: BootstrapComparison, top: int = -1, ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 8))
    cidf = comparison.get_confidence_intervals()
    cidf = cidf.sort_values("median", ascending=comparison.minimise)
    means = []
    yticks = []
    handles = []

    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", zorder=-1)

    n = len(cidf) if top <= 0 else min(len(cidf), top)
    for i, (algorithm, bounds) in enumerate(cidf.iloc[:n].iterrows()):
        pos = n - i
        means.append([bounds["median"], pos])
        yticks.append(algorithm)
        height = 0.5
        bar = patches.Rectangle(
            (bounds["lb"], pos - (height / 2)),
            bounds["ub"] - bounds["lb"],
            height,
            facecolor=(0, 0, 0.78, 0.4),
            label="{:.2f}% CI".format(1 - comparison.alpha),
        )
        p = ax.add_patch(bar)

    handles.append(ax.scatter(*zip(*means), color="red", label="Median", alpha=0.8))
    handles.append(p)

    ax.set_xlabel("Performance")
    ax.set_ylabel("Solver")
    ax.set_yticks(list(range(1, n + 1)))
    ax.set_yticklabels(yticks[::-1])
    ax.legend(handles=handles)
    if show:
        plt.tight_layout()
        plt.show()