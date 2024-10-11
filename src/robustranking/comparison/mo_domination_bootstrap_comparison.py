import itertools

import numpy as np
import pandas as pd

# Local imports
from robustranking.benchmark import Benchmark
from robustranking.comparison.bootstrap_comparison import BootstrapComparison
from robustranking.utils.multiobjective import dominates, incomparable


class MODominationBootstrapComparison(BootstrapComparison):
    """Multi-objective Bootstrap Comparison based on actual domination."""

    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: [dict | bool] = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int | np.random.RandomState] = 42):
        """
        Initialise function.

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
        super().__init__(benchmark, minimise, bootstrap_runs, alpha, aggregation_method, rng)

    def _get_distributions(self, always_minimise=True, lock=False, ranking=True) -> np.ndarray:
        if self._lock_dist:
            return self._dist_cache

        dist = super()._get_distributions(always_minimise=always_minimise)
        if not ranking:
            return dist

        rank_dist = 0.5 - np.ones(dist.shape[:2] + (1, ))
        for sample in range(self.bootstrap_runs):
            for a1, a2 in itertools.product(range(dist.shape[0]), repeat=2):
                # print(f"{dist[a2, sample, :]} vs {dist[a1, sample, :]}")
                # print(f"{a1} dominates {a2}: {dominates(dist[a2, sample, :], dist[a1, sample, :])}")
                # print(f"{a1} uncomparable {a2}: {uncomparable(dist[a2, sample, :], dist[a1, sample, :])}")
                # print(f"{rank_dist[a1, sample, 0]=}", end="")
                if dominates(dist[a2, sample, :], dist[a1, sample, :]):
                    rank_dist[a1, sample, 0] += 1
                elif incomparable(dist[a2, sample, :], dist[a1, sample, :]):
                    rank_dist[a1, sample, 0] += 0.5
                # print(f"-> {rank_dist[a1, sample, 0]}")

        if lock:
            self._lock_dist = True
            self._dist_cache = rank_dist

        return rank_dist

    def get_confidence_intervals(self, alpha: None | float = None) -> pd.DataFrame:
        """
        Computes the upper and lower bounds of the 1-alpha confidence interval.

        Returns:
            A pandas DataFrame with the bounds and the mean performance
        """
        cache = self._get_cache()
        distributions = self._get_distributions(always_minimise=False)
        meta_data = cache["meta_data"]

        alpha = self.alpha if alpha is None else alpha
        lower_bound = alpha / 2
        median = 0.5
        upper_bound = 1 - (alpha / 2)

        confidence_bounds = np.quantile(distributions, (median, lower_bound, upper_bound), axis=1)
        df = pd.DataFrame(confidence_bounds[:, :, 0].T, columns=["median", "lb", "ub"])
        df["algorithm"] = meta_data["algorithms"]
        # if len(meta_data["objectives"]) > 1:
        df["objective"] = "ranking_score"
        df = df.set_index(["algorithm", "objective"])

        return df
