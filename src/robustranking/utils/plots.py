import itertools
import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robustranking.comparison.abstract_comparison import \
        AbstractAlgorithmComparison
    from robustranking.comparison.bootstrap_comparison import \
        BootstrapComparison
    from robustranking.comparison.mo_bootstrap_comparison import \
        MOBootstrapComparison
    from robustranking.comparison.mo_domination_bootstrap_comparison import \
        MODominationBootstrapComparison
    import matplotlib as plt


# Bootstrap related plots
def plot_distribution(comparison: 'BootstrapComparison', algorithm: str, objective: str = None, ax: 'plt.axes' = None):
    """Plot kernel density estimates of the bootstrap distribution for an algorithm."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        raise ImportError("Function plot_distribution requires matplotlib to be installed.")

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
            color=(35 / 255, 134 / 255, 247 / 255),
            edgecolor=(23 / 255, 90 / 255, 166 / 255),
            alpha=0.5)

    histogram, edges = np.histogram(distribution, bins, density=True)
    kde = gaussian_kde(distribution)
    x = np.linspace(edges[0], edges[-1], 2048)
    ax.plot(x, kde(x), label="kernel density estimation")

    df = comparison.get_confidence_intervals()
    bounds = df.loc[(algorithm, objective), :]
    red = (237 / 255, 59 / 255, 43 / 255)
    ax.axvline(bounds["median"], color=red, linestyle="-", label="median ({:.0f})".format(bounds["median"]))
    ax.axvline(bounds["lb"],
               color=red,
               linestyle="--",
               label="{} CI bounds ({:.0f}, {:.0f})".format(1 - comparison.alpha, bounds["lb"], bounds["ub"]))
    ax.axvline(bounds["ub"], color=red, linestyle="--")

    ax.set_title(algorithm)
    ax.set_xlabel("Performance")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1.0))

    if show:
        plt.tight_layout()
        plt.show()


def plot_distributions_comparison(comparison: 'BootstrapComparison',
                                  algorithms: list,
                                  objective: str = None,
                                  show_p_values: bool = True,
                                  ax=None):
    """Plot the kernel density estimation of the bootstrap distribution for each algorithm."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        raise ImportError("Function plot_distributions_comparison requires matplotlib to be installed.")

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

        # CI shadow
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


def plot_ci_list(comparison: ['BootstrapComparison', 'MODominationBootstrapComparison'],
                 objective: str = None,
                 top: int = -1,
                 ax=None):
    """Plot the confidence intervals of the given comparison."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        from robustranking.comparison.mo_domination_bootstrap_comparison import MODominationBootstrapComparison
    except ImportError:
        raise ImportError("Function plot_ci_list requires matplotlib to be installed.")

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
        cidf = cidf.sort_values("median", ascending=True)  # Always minimise the ranking
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
    p = None
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
    if p is not None:
        handles.append(p)

    ax.set_xlabel("Performance")
    ax.set_ylabel("Solver")
    ax.set_yticks(list(range(1, n + 1)))
    ax.set_yticklabels(yticks[::-1])
    ax.legend(handles=handles)
    if show:
        plt.tight_layout()
        plt.show()


def plot_ci_density_estimations(comparison: 'MOBootstrapComparison',
                                algorithms: list | str = None,
                                show_names: bool | dict = False,
                                show_kde: bool = True,
                                show_contours: bool = True,
                                max_samples: float = 1000,
                                ax=None):
    """Plot the confidence areas of a multi-objective ranking."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        from scipy.stats import gaussian_kde
    except ImportError:
        raise ImportError("Function plot_ci_density_estimations requires matplotlib to be installed.")

    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(6, 6))

    cache = comparison._get_cache()
    performances = cache["distributions"]
    if performances.shape[1] > max_samples:
        performances = performances[:, :max_samples, :]
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
        xi, yi = np.mgrid[performances[algids, :, 0].min():performances[algids, :, 0].max():resolution * 1j,
                          performances[algids, :, 1].min():performances[algids, :, 1].max():resolution * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = (zi - zi.min()) / (zi.max() - zi.min())  # Normalize to be able to get quantiles

        colors = [(1, 1, 1, 0), end_colors[cid % len(end_colors)]]
        cmap1 = LinearSegmentedColormap.from_list("alpha", colors, N=256)

        if show_kde:
            ax.pcolormesh(xi,
                          yi,
                          zi.reshape(xi.shape),
                          shading="auto",
                          cmap=cmap1,
                          zorder=1,
                          norm=LogNorm(vmin=0.01),
                          alpha=0.66,
                          rasterized=True)
        if show_contours:
            levels = [0.05, 0.25, 0.5]  # 95% and 75%, 50% ci
            ax.contour(xi,
                       yi,
                       zi.reshape(xi.shape),
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
    comparisons: dict[str, 'AbstractAlgorithmComparison'],
    objective: str = None,
    line_color: [tuple[int, int, int, int] | str] = (0, 0, 0, 0.5),
    right_ticks: bool = False,
    hue=None,
    linestyle=None,
    ax=None,
    *args,
    **kwargs,
):
    """
    Plot a line rank plot based on a dictionary of rankings.

    The keys of the dict indicate the xticks.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Function plot_line_ranks requires matplotlib to be installed.")

    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots(figsize=(8, 8))

    # TODO: Check if algorithms are consistent across comparisons

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
            ax.set_yticks(list(range(1, 1 + n_algorithms)))
            ax.set_yticklabels(list(ranking.index)[::-1])
            ax.grid(False)
            ax.set_ylim(0.5, n_algorithms + .5)
        elif right_ticks and compid == len(comparisons) - 1:
            ax_r = ax.twinx()
            ax_r.set_yticks(list(range(1, 1 + n_algorithms)))
            ax_r.set_yticklabels(list(ranking.index)[::-1])
            ax_r.grid(False)
            ax_r.set_ylim(0.5, n_algorithms + .5)

        # Points
        colors = "blue"
        if isinstance(hue, dict):
            colors = [hue[solver] for solver in list(ranking.index)]
        elif hue is not None:
            colors = hue
        ax.scatter(np.ones(n_algorithms) * compid,
                   n_algorithms - np.arange(n_algorithms),
                   zorder=1000,
                   c=colors,
                   edgecolors='black')

        # Line administration
        for i, algname in enumerate(ranking.index):
            if algname not in lines:
                lines[algname] = list()
            lines[algname].append((compid, n_algorithms - i))

        # Group rectangles
        if "group" in ranking.columns:
            groups = ranking.groupby("group").size().sort_index()
            logging.debug(f"{groups=}")
            offset = n_algorithms
            for _, groupsize in groups.items():
                bar = patches.FancyBboxPatch(
                    (compid - 0.05, offset + 0.4),
                    0.1,
                    -(groupsize - .2),
                    facecolor=(0, 0, 0, 0.2),
                    boxstyle="Round4, pad=0, rounding_size=0.025",
                    # linestyle="--",
                    edgecolor="black",
                    # label="{:.2f}% CI".format(1 - comparison.alpha),
                    zorder=-1000,
                )
                if groupsize > 0:
                    ax.add_patch(bar)
                offset = offset - groupsize

    # Plot lines
    for algname, line in lines.items():
        line = np.array(line)
        style = dict()
        if isinstance(hue, dict):
            style["c"] = hue[algname]
        elif linestyle is not None:
            style["c"] = linestyle
        elif hue is not None:
            style["c"] = hue

        if isinstance(linestyle, dict):
            style["ls"] = linestyle[algname]
        elif linestyle is not None:
            style["ls"] = linestyle

        ax.plot(*line.T, **style)
        # Entries
        ax.plot((line[0, 0], line[0, 0] - 0.1), (line[0, 1], line[0, 1]), **style)
        ax.plot((line[-1, 0], line[-1, 0] + 0.1), (line[-1, 1], line[-1, 1]), **style)

    ax.set_xticks(list(range(len(comparisons))))
    ax.set_xticklabels(comparisons.keys())

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)

    if show:
        plt.tight_layout()
        plt.show()


def __compute_CD(avranks, n, alpha: float = 0.05):
    """
    Returns critical difference for Nemenyi test.

    Computer according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test is for Nemenyi two-tailed test.

    See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    """
    try:
        from scipy.stats import studentized_range
    except ImportError:
        raise ImportError()

    k = len(avranks)
    if k <= 1:
        return 0
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
    q_alpha /= np.sqrt(2)
    cd = q_alpha * (k * (k + 1) / (6.0 * n))**0.5
    return cd


def __graph_ranks(avranks,
                  names,
                  cd=None,
                  cdmethod=None,
                  lowv=None,
                  highv=None,
                  width=6,
                  textspace=1,
                  reverse=False,
                  filename=None,
                  ax=None,
                  title=None,
                  **kwargs):
    """
    Taken from https://github.com/biolab/orange3/blob/master/Orange/evaluation/scoring.py

    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(ln, n):
        """Returns only nth elemnt in a list."""
        n = lloc(ln, n)
        return [a[n] for a in ln]

    def lloc(ln, n):
        """List location in list of list structure.

        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(ln[0]) + n
        else:
            return n

    def mxrange(lr):
        """Multiple xranges.

        Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs if abs(sums[i] - sums[j]) <= hsd]

            # keep only longest

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis

    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(ln):
        return [a * hf for a in ln]

    def wfl(ln):
        return [a * wf for a in ln]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(ln, color='k', **kwargs):
        """Input is a list of pairs of points."""
        ax.plot(wfl(nth(ln, 0)), hfl(nth(ln, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    # Main line
    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    # Line ticks
    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    # Rank numbers
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    k = len(ssums)

    def namestring(name, rank):
        #         name = name if len(name) <= 7 else name[:6]+"..."
        return f"{name} ({rank:.1f})"

    # Left side rank names

    textline = min([rankpos(x) for x in ssums[:math.ceil(k / 2)]])
    #     print(textline)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2

        line(
            [(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textline - 0.5, chei)],
            # (textspace - 0.1, chei)],
            linewidth=0.7)
        # text(textspace - 0.2, chei, namestring(nnames[i], ssums[i]), ha="right", va="center")
        text(textline - 0.6, chei, namestring(nnames[i], ssums[i]), ha="right", va="center")

    # Right side rank names
    textline = max([rankpos(x) for x in ssums[math.ceil(k / 2):]])
    #     print(textline)
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line(
            [(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textline + 0.5, chei)],
            # (textspace + scalewidth + 0.1, chei)],
            linewidth=0.7)
        # text(textspace + scalewidth + 0.2, chei, namestring(nnames[i], ssums[i]), ha="left", va="center")
        text(textline + 0.6, chei, namestring(nnames[i], ssums[i]), ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        cddistance = cline - 0.325

        line([(begin, cddistance), (end, cddistance)], linewidth=0.7)
        line([(begin, cddistance + bigtick / 2), (begin, cddistance - bigtick / 2)], linewidth=0.7)
        line([(end, cddistance + bigtick / 2), (end, cddistance - bigtick / 2)], linewidth=0.7)
        text((begin + end) / 2, cddistance - 0.15, "CD", ha="center", va="top")

        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for ln, r in lines:
                line([(rankpos(ssums[ln]) - side, start), (rankpos(ssums[r]) + side, start)], linewidth=2.5)
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline + 1), (end, cline)], linewidth=2.5)
        line([(begin, cline + bigtick / 2), (begin, cline - bigtick / 2)], linewidth=2.5)
        line([(end, cline + bigtick / 2), (end, cline - bigtick / 2)], linewidth=2.5)

    if title:
        ax.set_title(title, y=0.9)

    if filename:
        # print_figure(fig, filename, **kwargs)
        plt.savefig(filename, bbox_inches="tight")


def plot_critical_difference(comparison: 'AbstractAlgorithmComparison', alpha=0.05, **kwargs):
    """Plots critical difference plot."""
    cache = comparison._get_cache()
    meta_data = cache["meta_data"]

    if len(meta_data["objectives"]) > 1:
        warnings.warn(f"More than one objectives detected! Only using the first one; '{meta_data['objectives'][0]}'")

    avranks = list(cache["aggregation"][:, 0])

    print("TEST")

    __graph_ranks(avranks,
                  meta_data["algorithms"],
                  cd=__compute_CD(avranks, n=len(meta_data["instances"]), alpha=alpha),
                  **kwargs)
