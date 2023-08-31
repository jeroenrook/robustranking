import itertools
import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Local imports
from robustranking.comparison.bootstrap_comparison import BootstrapComparison
from robustranking.benchmark import Benchmark
from robustranking.utils.multiobjective import fast_non_dominated_sorting, dominates


class MOBootstrapComparison(BootstrapComparison):
    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: [dict | bool] = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int | np.random.RandomState] = 42):
        """

        Args:
            benchmark: Benchmark class
            minimise: Whether it is the goal to minimise or maximise the objectives.
                Can be different for each objective.
            bootstrap_runs: the number of bootstrap samples that should be generated
            alpha: the alpha value
            aggregation_method: a 1-D aggregation function. Default is np.mean. Can be
                different for each objective.
            rng: Random number generator.
        """
        super().__init__(benchmark,
                         minimise,
                         bootstrap_runs,
                         alpha,
                         aggregation_method,
                         rng)

    def _statistical_test(self, s1: int, s2: int) -> float:
        """
        Performs a statistical test on the null hypothesis that algorithm 1 (s1) is equal
        or worse that algorithm 2 (s2).
        Args:
            s1:
            s2:

        Returns:
            p-value of the test. If this value is below the alpha value, then the
            hypothesis is rejected and s1 is better than s2.
        """
        cache = self._get_cache()
        dist = copy.copy(cache["distributions"])

        # Always minimise
        if isinstance(self.minimise, dict):
            for i, obj in enumerate(cache["meta_data"]["objectives"]):
                if self.minimise[obj] is False:
                    dist[:, :, i] = -1 * dist[:, :, i]
        elif self.minimise is False:
            dist = -1*dist

        # H0: s1 is dominated by s2
        # This means that incomparable solutions can be statistically untied
        # wins = [dominates(*p) for p in zip(dist[s2, :, :], dist[s1, :, :])]
        # wins = np.count_nonzero(wins)

        # H0: s1 is dominated by of incomparable with s2
        wins = [dominates(*p) for p in zip(dist[s1, :, :], dist[s2, :, :])]
        wins = self.bootstrap_runs - np.count_nonzero(wins)

        p_value = wins / self.bootstrap_runs  # p-value

        # p_value < self.alpha -> reject -> s1 is better performing than s2
        # p_value >= self.alpha -> accept -> s1 is equal or worse performing than s2
        return p_value

    def get_ranking(self, visualise: bool = False) -> pd.DataFrame:
        """
        Generates a robust ranking which groups statistically equal algorithm together.
        Returns:
            Dataframe with the algorithm as index and a columns with the rank (group) and a column with the mean
            performance over all the bootstrap samples
        """
        cache = self._get_cache()
        dist = copy.copy(cache["distributions"])
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        groups = {}
        groupid = 0

        # Always minimise
        if len(meta_data["objectives"]) > 1 and isinstance(self.minimise, dict):
            for oid, o in enumerate(meta_data["objectives"]):
                dist[:, :, oid] *= 1 if self.minimise[o] else -1  # flip when maximise
        elif not self.minimise:
            dist *= -1  # flip

        replace = np.inf  # if self.minimise else -np.inf
        argfunc = np.argmin  # if self.minimise else np.argmax

        # fractional_wins = np.zeros(n_algorithms, dtype=float)
        # algos, wins = np.unique(argfunc(dist, axis=0), return_counts=True)
        # fractional_wins[algos] = wins
        # fractional_wins = fractional_wins / self.bootstrap_runs

        candidates_mask = np.ones(n_algorithms, dtype=bool)
        while np.count_nonzero(candidates_mask) > 0:
            # Find the winner amongst the remaining candidates
            # Pick the algorithm that over all has the lowest average front
            #TODO make get_winner class since ranking is the same as with normal bootstrap
            if np.count_nonzero(candidates_mask) > 1:
                allranks = []
                for sample in range(dist.shape[1]):
                    _, _, _, ranks = fast_non_dominated_sorting(dist[candidates_mask, sample, :])
                    allranks.append(ranks)
                meanranks = np.mean(allranks, axis=0)
                # print(f"{meanranks=}")
                winner = np.argwhere(candidates_mask).flatten()[np.argmin(meanranks)]
            else:
                winner = np.argwhere(candidates_mask).flatten()[0]
                meanranks = np.array([0])
            # print(f"{winner=}")

            if visualise:
                labels = {c: meta_data["algorithms"][c] for c in range(n_algorithms)}
                for c, p in zip(np.argwhere(candidates_mask).flatten(), meanranks):
                    labels[c] += f"\n~front={p:.3f}"

            groups[groupid] = [(winner, np.mean(dist[winner, :, :], axis=0))]
            candidates_mask[winner] = False  # Remove winner from candidates
            # dist[winner, :, :] = replace #TODO handle mixed directions
            if np.count_nonzero(candidates_mask) == 0:
                break

            # Perform statistical tests to find candidates that are statistically tied
            candidates = np.argwhere(candidates_mask).flatten()
            pvalues = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                pvalues[i] = self._statistical_test(winner, candidate)  # H0: winner <= candidate
            # Multiple test correction
            pvalues_order = np.argsort(pvalues)

            # Holm-Bonferroni
            reject = np.zeros(len(candidates), dtype=bool)  # Do not reject any test by default
            for i, index in enumerate(pvalues_order):
                corrected_alpha = self.alpha / (len(candidates) - i)  # Holm-Bonferroni
                if pvalues[index] < corrected_alpha:
                    # Reject H0 -> winner > candidate
                    reject[index] = True
                else:
                    break

            if visualise:
                # TODO check for dimensions <= 2
                for c, p in zip(candidates, pvalues):
                    labels[c] += f"\np={p:.3f}"

                fig, ax = self._plot_state(
                    points=np.mean(dist, axis=1),
                    winner=winner,
                    inactive=np.argwhere(~candidates_mask).flatten(),
                    labels=labels,
                )
                plt.title(f"Round {groupid+1}")
                plt.tight_layout()
                plt.show()

            # Not rejecting means they are statistically tied
            ties = candidates[~reject]
            for candidate in ties:
                groups[groupid].append((candidate, 0))
                candidates_mask[candidate] = False
                # dist[candidate, :, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            # group wins
            # group_wins = np.zeros(len(algorithms), dtype=float)
            # algos, wins = np.unique(argfunc(dist[algorithms, :, :], axis=0), return_counts=True)
            # group_wins[algos] = wins
            # group_wins = group_wins / self.bootstrap_runs
            # algmap = {a: i for i, a in enumerate(algorithms)}

            for (algorithm, performance) in algorithms:
                results.append({"algorithm": meta_data["algorithms"][algorithm],
                                "group": group + 1,
                                #"ranked 1st": fractional_wins[algorithm],
                                #"group wins": group_wins[algmap[algorithm]]
                                })

        df = pd.DataFrame(results).set_index("algorithm").sort_values(
            ["group"], ascending=[True])
        #df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        return df

    def _plot_state(self,
                    points: np.ndarray,
                    winner: int = None,
                    inactive: list[int] = None,
                    labels: dict[str] = None):
        fig, ax = plt.subplots(1,1, figsize=(7, 5))

        meta_data = self._get_cache()["meta_data"]
        ax.set_xlabel(meta_data["objectives"][0])
        ax.set_ylabel(meta_data["objectives"][1])

        if isinstance(self.minimise, dict):
            for i, o in enumerate(meta_data["objectives"]):
                points[:, i] *= 1 if self.minimise[o] else -1
        elif not self.minimise:
            points *= -1

        color = [(0, 0, 1, 1) for _ in range(len(points))]
        if inactive is not None:
            for i in inactive:
                color[i] = (0, 0, 0, 0.25)
        if winner is not None:
            color[winner] = (1, 0, 0, 1)

        sc = plt.scatter(*zip(*points), c=color)

        if labels is not None:
            # for i, text in labels.items():
            #     plt.text(points[i, 0], points[i, 1], text, clip_on=True)
            annot = ax.annotate("",
                                xy=(0, 0),
                                xytext=(10, 10),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = ""
                for n in ind['ind']:
                    if n in labels:
                        text += labels[n] + "\n"
                annot.set_text(text)
                annot.get_bbox_patch().set_facecolor((0, 0, 0, 0.2))
                annot.get_bbox_patch().set_alpha(0.5)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

        return fig, ax
