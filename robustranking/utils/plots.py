import itertools
import logging
from typing import Sequence

from scipy.stats import gaussian_kde

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from robustranking.comparison import BootstrapComparison, MOBootstrapComparison, MODominationBootstrapComparison
from robustranking.comparison.abstract_comparison import AbstractAlgorithmComparison


# Bootstrap related plots
def plot_distribution(comparison: BootstrapComparison,
                      algorithm: str,
                      objective: str = None,
                      ax: plt.axes = None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(9, 6))

    cache = comparison._get_cache()
    meta_data = cache["meta_data"]
    if objective is None:
        if len(meta_data["objectives"]) > 1:
            print(f"Please select an objective to make the list for: {meta_data['objectives']}")
            return
        objective = meta_data["objectives"][0]

    obj_id = meta_data["objectives"].index(objective)

    distribution = comparison.get_bootstrap_distribution(algorithm)
    distribution = distribution[:, obj_id]
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
    bounds = df.loc[(algorithm, objective), :]
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
                                  objective: str = None,
                                  show_p_values: bool = True,
                                  ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(9, 6))

    cache = comparison._get_cache()
    meta_data = cache["meta_data"]
    if objective is None:
        if len(meta_data["objectives"]) > 1:
            print(f"Please select an objective to make the list for: {meta_data['objectives']}")
            return
        objective = meta_data["objectives"][0]

    obj_id = meta_data["objectives"].index(objective)

    distributions = []
    for algorithm in algorithms:
        distributions.append(comparison.get_bootstrap_distribution(algorithm)[:, obj_id])

    distributions = np.stack(distributions)

    df = comparison.get_confidence_intervals()
    df = df[df.index.isin([objective], level=1)]

    for i, algorithm in enumerate(algorithms):
        kde = gaussian_kde(distributions[i, :])
        x = np.linspace(np.min(distributions), np.max(distributions), 2048)
        ax.fill_between(x, kde(x), label=algorithm, alpha=0.66, edgecolor="black")

        #CI shadow
        x = np.linspace(df.loc[(algorithm, objective), "lb"], df.loc[(algorithm, objective), "ub"], 2048)
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


def plot_ci_list(comparison: BootstrapComparison | MODominationBootstrapComparison,
                 objective: str = None,
                 top: int = -1,
                 ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 8))
    cidf = comparison.get_confidence_intervals()

    cache = comparison._get_cache()
    meta_data = cache["meta_data"]
    if objective is None:
        if len(meta_data["objectives"]) > 1:
            print(f"Please select an objective to make the list for: {meta_data['objectives']}")
            return
        objective = meta_data["objectives"][0]

    cidf = cidf[cidf.index.isin([objective], level=1)]
    if isinstance(comparison, MODominationBootstrapComparison):
        cidf = cidf.sort_values("median", ascending=True) #Always minimise the ranking
    elif isinstance(comparison.minimise, dict):
        cidf = cidf.sort_values("median", ascending=comparison.minimise[objective])
    else:
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
        yticks.append(algorithm[0])
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


def plot_ci_density_estimations(
        comparison: MOBootstrapComparison,
        algorithms: list | str = None,
        show_names: bool | dict = False,
        show_kde: bool = True,
        show_contours: bool = True,
        max_samples: float = 1000,
        ax=None):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(6, 6))

    cache = comparison._get_cache()
    performances = cache["distributions"]
    if performances.shape[1] > max_samples:
        performances = performances[:,:max_samples,:]
    meta_data = cache["meta_data"]

    end_colors = [
        (0, 0, 1, 0.8),
        (0, 1, 0, 0.8),
        (1, 0, 0, 0.8),
        (0, 1, 1, 0.8),
        (1, 0, 1, 0.8),
        (1, 1, 0, 0.8),
    ]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    end_colors = prop_cycle.by_key()['color']

    if isinstance(algorithms, str):
        algorithms = [algorithms]

    algids = [a in algorithms for a in meta_data["algorithms"]]
    for cid, algname in enumerate(algorithms):
        alg = meta_data["algorithms"].index(algname)
        x = performances[alg, :, 0]
        y = performances[alg, :, 1]
        k = gaussian_kde([x, y])
        resolution = 256
        xi, yi = np.mgrid[
                 performances[algids, :, 0].min():performances[algids, :, 0].max():resolution * 1j,
                 performances[algids, :, 1].min():performances[algids, :, 1].max():resolution * 1j
                 ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - zi.min()) / (zi.max() - zi.min())  # Normalize to be able to get quantiles

        colors = [
            (1, 1, 1, 0),
            end_colors[cid % len(end_colors)]
        ]
        cmap1 = LinearSegmentedColormap.from_list("alpha", colors, N=256)

        if show_kde:
            ax.pcolormesh(xi, yi, zi.reshape(xi.shape),
                          shading="auto",
                          cmap=cmap1,
                          zorder=1,
                          norm=LogNorm(vmin=0.01),
                          alpha=0.66,
                          rasterized=True)
        if show_contours:
            levels = [0.05, 0.25, 0.5]  # 95% and 75%, 50% ci
            ax.contour(xi, yi, zi.reshape(xi.shape),
                        levels=levels,
                        colors=len(levels) * [end_colors[cid % len(end_colors)]],
                        zorder=2,
                        alpha=0.5)
        if isinstance(show_names, dict):
            plt.text(np.mean(x), np.mean(y), f"{show_names[algname]}", zorder=30, ha="center", va="bottom", c="black")
        elif show_names:
            plt.text(np.mean(x), np.mean(y), f"{algname}", zorder=30, ha="center", va="bottom", c="black")

    # for s1, s2 in itertools.product(algorithms, repeat=2):
    #     print(f"H0: {s1:24} is dominated by or incomparable {s2:24}: p-value={comparison.statistical_test(s1, s2)}")

    ax.scatter(*zip(*(np.mean(performances[algids], axis=1).tolist())), c="black", alpha=0.8, zorder=3)
    ax.set_xlabel(meta_data["objectives"][0])
    ax.set_ylabel(meta_data["objectives"][1])

    # plt.grid()
    if show:
        plt.tight_layout()
        plt.show()

def plot_line_ranks(
        comparisons: dict[str, AbstractAlgorithmComparison],
        objective: str = None,
        line_color: [tuple[int, int, int, int] | str] = (0, 0, 0, 0.5),
        ax=None,
):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 8))

    # TODO: Check is algorithms are consistent across comparisons
    width, height = ax.figure.get_size_inches()

    lines = dict()

    for compid, (compname, comparison) in enumerate(comparisons.items()):
        logging.debug(f"{compid:2}: {compname}")

        cache = comparison._get_cache()
        meta_data = cache["meta_data"]
        if objective is None:
            if len(meta_data["objectives"]) > 1:
                print(f"Please select an objective to make the list for: {meta_data['objectives']}")
                return
            objective = meta_data["objectives"][0]

        ranking = comparison.get_ranking()
        n_algorithms = len(ranking)

        if compid == 0:
            ax.set_yticks(list(range(1, 1+n_algorithms)))
            ax.set_yticklabels(list(ranking.index)[::-1])

        # Points
        ax.scatter(np.ones(n_algorithms)*compid, n_algorithms-np.arange(n_algorithms), zorder=1000, c="blue")

        # Line administration
        for i, algname in enumerate(ranking.index):
            if algname not in lines:
                lines[algname] = list()
            lines[algname].append((compid, n_algorithms-i))

        # Group rectangles
        if "group" in ranking.columns:
            groups = ranking.groupby("group").size().sort_index()
            logging.debug(f"{groups=}")
            offset = n_algorithms
            for _, groupsize in groups.items():
                bar = patches.FancyBboxPatch(
                        (compid-0.05, offset+0.4),
                        0.1,
                        -(groupsize-.2),
                        facecolor=(0, 0, 0, 0.2),
                        boxstyle="Round4, pad=0, rounding_size=0.025",
                        # linestyle="--",
                        edgecolor="black",
                        # label="{:.2f}% CI".format(1 - comparison.alpha),
                        zorder=-1000,
                    )
                if groupsize > 0:
                    p = ax.add_patch(bar)
                offset = offset - groupsize

    for algname, line in lines.items():
        line = np.array(line)
        ax.plot(*line.T, c=line_color)
        #entries
        ax.plot((line[0, 0], line[0, 0]-0.1), (line[0, 1], line[0, 1]), c=line_color)
        ax.plot((line[-1, 0], line[-1, 0]+0.1), (line[-1, 1], line[-1, 1]), c=line_color)


    ax.set_xticks(list(range(len(comparisons))))
    ax.set_xticklabels(comparisons.keys())

    for side in ["left", "right","top", "bottom"]:
        ax.spines[side].set_visible(False)

    if show:
        plt.tight_layout()
        plt.show()
