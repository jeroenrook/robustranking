import copy
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustranking.benchmark import Benchmark
from robustranking.comparison.bootstrap_comparison import BootstrapComparison
from robustranking.utils.multiobjective import (dominates, fast_non_dominated_sorting)


class MOBootstrapComparison(BootstrapComparison):
    """Multi-objective boostrap comparison"""

    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: [dict | bool] = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int | np.random.RandomState] = 42,
                 winner_threshold: float = 0.50):
        """
        Initialize

        Args:
            benchmark: Benchmark class
            minimise: Whether it is the goal to minimise or maximise the objectives.
                Can be different for each objective.
            bootstrap_runs: the number of bootstrap samples that should be generated
            alpha: the alpha value
            aggregation_method: a 1-D aggregation function. Default is np.mean. Can be
                different for each objective.
            rng: Random number generator.
            winner_threshold: The ratio of being in the first dominated layer to be considered a winner.
        """
        super().__init__(benchmark, minimise, bootstrap_runs, alpha, aggregation_method, rng)

        self.winner_threshold = winner_threshold

    def _statistical_test(self, s1: int, s2: int) -> float:
        """
        Performs a statistical test on the null hypothesis that algorithm 1 (s1) is dominated by algorithm 2 (s2).

        Args:
            s1:
            s2:

        Returns:
            p-value of the test. If this value is below the alpha value, then the
            hypothesis is rejected and s1 is better than s2.
        """
        dist = self._get_distributions(always_minimise=True)

        # H0: s1 is dominated by s2
        # This means that incomparable solutions can be statistically untied
        # wins = [dominates(*p) for p in zip(dist[s2, :, :], dist[s1, :, :])]
        # wins = np.count_nonzero(wins)

        # H0: s1 is dominated by s2
        wins = [dominates(*p) for p in zip(dist[s2, :, :], dist[s1, :, :])]
        wins = np.count_nonzero(wins)
        # wins = self.bootstrap_runs - np.count_nonzero(wins)

        p_value = wins / self.bootstrap_runs  # p-value

        # p_value < self.alpha -> reject -> s1 is better performing than s2
        # p_value >= self.alpha -> accept -> s1 is equal or worse performing than s2
        return p_value

    def _probability_dominates(self, s1, s2) -> float:
        # Hughes 2001
        cache = self._get_cache()
        dist = copy.copy(cache["distributions"])

        # Always minimise TODO make helper function
        if isinstance(self.minimise, dict):
            for i, obj in enumerate(cache["meta_data"]["objectives"]):
                if self.minimise[obj] is False:
                    dist[:, :, i] = -1 * dist[:, :, i]
        elif self.minimise is False:
            dist = -1 * dist

        probs = []
        for o in range(len(cache["meta_data"]["objectives"])):
            p = sum([p[0] < p[1] for p in zip(dist[s1, :, o], dist[s2, :, o])]) / self.bootstrap_runs
            probs.append(p)

        return np.product(probs)

    def _probability_indifferent(self, s1, s2) -> float:
        # Hughes 2001
        return 1 - (self._probability_dominates(s1, s2) + self._probability_dominates(s2, s1))

    def hughes_ranking(self) -> pd.DataFrame:
        """Hughes 2001 probabilistic ranking"""
        cache = self._get_cache()
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        ranks = np.ones(n_algorithms) * -0.5
        for i, j in itertools.product(range(n_algorithms), repeat=2):
            ranks[i] += self._probability_dominates(j, i) + 0.5 * self._probability_indifferent(j, i)

        return pd.DataFrame(list(zip(meta_data["algorithms"], ranks)), columns=["Algorithm", "Rank"])

    def get_ranking(self, visualise: bool = False) -> pd.DataFrame:
        """
        Generates a robust ranking which groups statistically equal algorithm together.

        Returns:
            Dataframe with the algorithm as index and a columns with the rank (group) and a column with the mean
            performance over all the bootstrap samples
        """
        cache = self._get_cache()
        dist = self._get_distributions(always_minimise=True, lock=True)
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        groups = {}
        groupid = 0

        global_ranking = None  # Store the ND ranking of all solvers here for the stats afterwards

        candidates_mask = np.ones(n_algorithms, dtype=bool)
        group_wins = np.zeros(n_algorithms, dtype=float)
        while np.count_nonzero(candidates_mask) > 0:
            logging.info(f"Round {groupid}")
            logging.debug(f"{candidates_mask=}")
            # Find the winner amongst the remaining candidates
            # Pick the algorithm that over all has the lowest average front
            # TODO make get_winner function since ranking is the same as with normal bootstrap
            if np.count_nonzero(candidates_mask) > 1:

                active_candidates = np.argwhere(candidates_mask).flatten()

                allranks = np.zeros((len(active_candidates), self.bootstrap_runs))
                for sample in range(dist.shape[1]):
                    _, _, _, ranks = fast_non_dominated_sorting(dist[candidates_mask, sample, :])
                    allranks[:, sample] = ranks

                if global_ranking is None:
                    global_ranking = np.copy(allranks)

                in_first_layer = np.count_nonzero(allranks == 0, axis=1)
                for a, c in zip([meta_data['algorithms'][a] for a in active_candidates], in_first_layer):
                    logging.debug(f"{a:30} {c}")

                winners = np.where(in_first_layer > np.max(in_first_layer) * self.winner_threshold)[0]
                winners = [active_candidates[winner] for winner in winners]

                for rankpos, candidate in enumerate(active_candidates):
                    # Register the number of times a candidate was non-dominated among the others
                    group_wins[candidate] = np.count_nonzero(np.array(allranks)[:, rankpos] == 0)
            else:
                logging.info("Only one candidate left..")
                winners = np.argwhere(candidates_mask).flatten()
                group_wins[winners] = 1

            if visualise:
                labels = {c: f"({c}) {meta_data['algorithms'][c]}" for c in range(n_algorithms)}
                for c, p in zip(np.argwhere(candidates_mask).flatten(), in_first_layer / self.bootstrap_runs):
                    labels[c] += f"\n~front={p:.2f}"

            # winner = np.random.choice(winners)

            groups[groupid] = []
            for winner in winners:
                groups[groupid].append((winner, np.mean(dist[winner, :, :], axis=0), set()))
                candidates_mask[winner] = False  # Remove winner from candidates

            if np.count_nonzero(candidates_mask) == 0:
                print("No candidates to compare")
                break

            logging.info(f"{len(winners)} winners found: {[meta_data['algorithms'][w] for w in winners]}")

            allties = dict()
            for winner in winners:
                # Perform statistical tests to find candidates that are statistically tied
                candidates = np.argwhere(candidates_mask).flatten()
                pvalues = np.zeros(len(candidates))
                for i, candidate in enumerate(candidates):
                    pvalues[i] = self._statistical_test(winner, candidate)  # H0: candidate dominates the winner
                    logging.info(f"\t> {meta_data['algorithms'][winner]} is dominated by "
                                 f"{meta_data['algorithms'][candidate]} {pvalues[i]:.3%} times.")

                # Multiple test correction: Holm-Bonferroni
                pvalues_order = np.argsort(pvalues)
                reject = np.zeros(len(candidates), dtype=bool)  # Do not reject any test by default
                for i, index in enumerate(pvalues_order):
                    corrected_alpha = self.alpha / (len(candidates) - i)  # Holm-Bonferroni
                    if pvalues[index] < corrected_alpha:
                        # Reject H0 -> winner dominates candidate
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
                    if candidate not in allties:
                        allties[candidate] = set()
                    allties[candidate].add(winner)
                    # groups[groupid].append((candidate, 0)) #TODO get ND rank
                    # candidates_mask[candidate] = False
                    # dist[candidate, :, :] = replace

            for candidate, tiedwith in allties.items():
                groups[groupid].append((candidate, 0, tiedwith))  # TODO get ND rank
                candidates_mask[candidate] = False

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            # group wins
            # group_wins = np.zeros(len(algorithms), dtype=float)
            # algos, wins = np.unique(argfunc(dist[algorithms, :, :], axis=0), return_counts=True)
            # group_wins[algos] = wins
            # group_wins = group_wins / self.bootstrap_runs
            # algmap = {a: i for i, a in enumerate(algorithms)}

            for (algorithm, performance, ties) in algorithms:
                results.append({
                    "id": algorithm,
                    "algorithm": meta_data["algorithms"][algorithm],
                    "group": group + 1,
                    "winner": len(ties) == 0,
                    "ties": ties,
                    # "ranked 1st": fractional_wins[algorithm],
                    # "group wins": group_wins[algorithm] / self.bootstrap_runs,
                    "nd_rank_mean": np.mean(global_ranking[algorithm, :]),
                    "nd_rank_median": np.median(global_ranking[algorithm, :]),
                    "nd_rank_ci_lb": np.quantile(global_ranking[algorithm, :], self.alpha / 2),
                    "nd_rank_ci_ub": np.quantile(global_ranking[algorithm, :], 1 - self.alpha / 2),
                })

        df = pd.DataFrame(results).set_index("id", ).sort_values(["group", "winner", "nd_rank_median"],
                                                                 ascending=[True, False, True])
        # df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        self._unlock_distribution()

        return df

    def _plot_state(self, points: np.ndarray, winner: int = None, inactive: list[int] = None, labels: dict[str] = None):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

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
            for i, text in labels.items():
                plt.text(points[i, 0], points[i, 1], text, clip_on=True, verticalalignment='center')
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
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
