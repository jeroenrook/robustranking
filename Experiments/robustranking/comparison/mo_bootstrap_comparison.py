import itertools
import copy
import pandas as pd
import numpy as np

# Local imports
from robustranking.comparison.bootstrap_comparison import BootstrapComparison
from robustranking.benchmark import Benchmark
from robustranking.utils.multiobjective import fast_non_dominated_sorting


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
        super().__init__(benchmark, minimise, bootstrap_runs, alpha, aggregation_method,
                         rng)

    @staticmethod
    def _dominates(x, y):
        """
            assumes minimising
        Args:
            x: list of objective vector
            y: list of objective vector

        Returns: Bool which says if x dominates y (x > y)

        """

        return np.count_nonzero(x >= y) == 0 and np.count_nonzero(x < y) > 0

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
        dist = cache["distributions"]

        #TODO dominance


        #TODO handle non-numeric objectives
        if isinstance(self.minimise, dict):
            for i, obj in enumerate(cache["meta_data"]["objectives"]):
                if self.minimise[obj] is False:
                    dist[:, :, i] = -1 * dist[:, :, i]
        elif self.minimise is False:
            dist = -1*dist
        #TODO how to handle uncomparable situations?
        wins = [self._dominates(*p) for p in zip(dist[s2, :, :], dist[s1, :, :])]
        wins = np.count_nonzero(wins)

        p_value = wins / self.bootstrap_runs  # p-value

        # p_value < self.alpha -> reject -> s1 is better performing than s2
        # p_value >= self.alpha -> accept -> s1 is equal or worse performing than s2
        return p_value

    def get_ranking(self) -> pd.DataFrame:
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

        replace = np.inf if self.minimise else -np.inf
        argfunc = np.argmin if self.minimise else np.argmax

        fractional_wins = np.zeros(n_algorithms, dtype=float)
        algos, wins = np.unique(argfunc(dist, axis=0), return_counts=True)
        fractional_wins[algos] = wins
        fractional_wins = fractional_wins / self.bootstrap_runs

        candidates_mask = np.ones(n_algorithms, dtype=bool)
        while np.count_nonzero(candidates_mask) > 0:
            # Find the winner amongst the remaining candidates
            # Pick the algorithm that over all samples dominates the most algorithms

            if np.count_nonzero(candidates_mask) > 1:
                domination_count = np.zeros(np.count_nonzero(candidates_mask))
                for sample in range(dist.shape[1]):
                    _, dl, _, _ = fast_non_dominated_sorting(dist[candidates_mask, sample, :])
                    domination_count += [len(d) for d in dl]
                print(domination_count)
                winner = np.argwhere(candidates_mask).flatten()[np.argmax(domination_count)]
            else:
                winner = np.argwhere(candidates_mask).flatten()[0]
            print(f"{winner=}")

            # distwins = argfunc(distributions, axis=0)
            # algos, wins = np.unique(distwins, return_counts=True)
            # winner = algos[np.argmax(wins)]

            groups[groupid] = [(winner, np.mean(dist[winner, :]))]
            candidates_mask[winner] = False  # Remove winner from candidates
            # dist[winner, :, :] = replace #TODO handle mixed directions
            if np.count_nonzero(candidates_mask) == 0:
                break

            # Perform statistical tests to find candidates that are statistically tied
            candidates = np.argwhere(candidates_mask).flatten()
            pvalues = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                pvalues[i] = self._statistical_test(winner,
                                                    candidate)  # H0: winner <= candidate
            # Multiple test correction
            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            pvalues_order = np.argsort(pvalues)
            # reject = pvalues < self.alpha  # no correction
            # reject = multipletests(pvalues, self.alpha, method="holm")[0]  # hommel

            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            # Holm-Bonferroni
            reject = np.zeros(len(candidates),
                              dtype=bool)  # Do not reject any test by default
            for i, index in enumerate(pvalues_order):
                corrected_alpha = self.alpha / (len(candidates) - i)  # Holm-Bonferroni
                if pvalues[index] < corrected_alpha:
                    # Reject H0 -> winner > candidate
                    reject[index] = True
                else:
                    break

            # Not rejecting means they are statistically tied
            ties = candidates[~reject]
            for candidate in ties:
                groups[groupid].append((candidate, 0))
                candidates_mask[candidate] = False
                dist[candidate, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            #group wins
            group_wins = np.zeros(len(algorithms), dtype=float)
            algos, wins = np.unique(argfunc(dist[algorithms, :], axis=0), return_counts=True)
            group_wins[algos] = wins
            group_wins = group_wins / self.bootstrap_runs
            algmap = {a: i for i, a in enumerate(algorithms)}

            for (algorithm, performance) in algorithms:
                results.append({"algorithm": meta_data["algorithms"][algorithm],
                                "group": group + 1,
                                "ranked 1st": fractional_wins[algorithm],
                                "group wins": group_wins[algmap[algorithm]]})

        df = pd.DataFrame(results).set_index("algorithm").sort_values(
            ["group", "ranked 1st"], ascending=[True, False])
        df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        return df