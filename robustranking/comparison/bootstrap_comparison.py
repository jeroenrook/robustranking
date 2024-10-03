import copy
import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import gaussian_kde
from typing_extensions import Self

from robustranking.benchmark import Benchmark
# Local imports
from robustranking.comparison.abstract_comparison import \
    AbstractAlgorithmComparison


class BootstrapComparison(AbstractAlgorithmComparison):
    """Comparing algorithms based on bootstrap resampling of the instances."""

    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: bool | dict = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int | np.random.RandomState] = 42):
        """
        Bootstrap comparison.

        Args:
            benchmark: Benchmark class
            minimise: Whether it is the goal to minimise or maximise the objectives
            bootstrap_runs: the number of bootstrap samples that should be generated
            alpha: the alpha value
            aggregation_method: a 1-D aggregation function. Default is np.mean
            rng: Random number generator.
        """
        super().__init__(benchmark, minimise)

        self.bootstrap_runs = bootstrap_runs
        self.alpha = alpha
        self.aggregation_method = aggregation_method

        if isinstance(rng, int):
            self.rng = np.random.default_rng(rng)  # Use this to generate randomness
        else:
            self.rng = rng

        self._lock_dist = False
        self._dist_cache = None

    def _get_samples(self, num_instances: int, bootstrap_runs: int | None = None) -> np.ndarray:
        """
        Generates the samples.

        Args:
            num_instances: number of instances.
            bootstrap_runs: number of bootstrap samples.

        Returns:
            samples
        """
        bootstrap_runs = self.bootstrap_runs if bootstrap_runs is None else bootstrap_runs

        if binom(2 * num_instances, num_instances) <= bootstrap_runs:
            warnings.warn(f"There are only {binom(2*num_instances, num_instances):.0f} unique samples possible,  "
                          f"which is less than the requested {bootstrap_runs} bootstrap samples. Duplicate samples are "
                          f"inevitable. "
                          f"Consider increasing the number of instances or reducing the number of bootstraps.")

        return self.rng.choice(
            np.arange(0, num_instances),
            size=(num_instances, bootstrap_runs),
            replace=True,
        )

    def _statistical_test(self, s1: int, s2: int) -> float:
        """
        Statistical test helper function

        Performs a statistical test on the null hypothesis that algorithm 1 (s1) is equal
        or worse that algorithm 2 (s2).

        Args:
            s1:
            s2:

        Returns:
            p-value of the test. If this value is below the alpha value, then the
            hypothesis is rejected and s1 is better than s2.
        """
        distributions = self._get_distributions(always_minimise=True)

        # if self.minimise:
        wins = np.count_nonzero(distributions[s1, :] >= distributions[s2, :])
        # else:
        # wins = np.count_nonzero(distributions[s1, :] <= distributions[s2, :])

        p_value = wins / self.bootstrap_runs  # p-value

        # p_value < self.alpha -> reject -> s1 is better performing than s2
        # p_value >= self.alpha -> accept -> s1 is equal or worse performing than s2
        return p_value

    def statistical_test(self, algorithm1: str, algorithm2: str) -> float:
        """
        Statistical test.

        Performs a statistical test on the null hypothesis that algorithm 1 (s1) is equal
        or worse that algorithm 2 (s2).

        Args:
            algorithm1: Name of the algorithm
            algorithm2: Name of the algorithm

        Returns:
            p-value of the test. If this value is below the alpha value, then the
            hypothesis is rejected and algorithm1 is better than algorithm2.
        """
        cache = self._get_cache()
        algorithms = cache["meta_data"]["algorithms"]
        s1 = algorithms.index(algorithm1)
        s2 = algorithms.index(algorithm2)
        return self._statistical_test(s1, s2)

    def compute(self) -> Self:
        """
        Compute the bootstrap samples.

        First the bootstrap samples are created. Then it collects for each algorithm
        the aggregated performance from each bootstrap sample.
        """
        if not self.benchmark.check_complete():
            raise ValueError("Benchmark table is not complete. Cannot compute.")
        array, meta_data = self.benchmark.to_numpy()

        if len(array.shape) == 2:
            array = array.reshape(-1, -1, 1)
        if len(array.shape) != 3:
            raise "Dimension error"

        # Generate n bootstrap samples from the instances
        bootstraps = self._get_samples(len(meta_data["instances"]))

        # Compute the performance of the algorithm on each bootstrap sample
        distributions = np.zeros((len(meta_data["algorithms"]), self.bootstrap_runs, len(meta_data["objectives"])))

        for alg, obj in itertools.product(range(array.shape[0]), range(array.shape[2])):
            performance = array[alg, :, obj]
            samples = np.take(performance, bootstraps)
            # Compute aggregated performance
            agg_method = self.aggregation_method
            if isinstance(self.aggregation_method, dict):
                agg_method = self.aggregation_method[meta_data["objectives"][obj]]

            distributions[alg, :, obj] = np.apply_along_axis(agg_method, 0, samples)

        self._cache = {
            "array": array,
            "bootstraps": bootstraps,
            "distributions": distributions,
            "meta_data": meta_data,
        }

        return self

    def _unlock_distribution(self):
        self._lock_dist = False
        self._dist_cache = None

    def _get_distributions(self, always_minimise=True, lock=False, **kwargs) -> np.ndarray:
        # TODO check of lock request while still lock -> yield error/warning
        if self._lock_dist:
            return self._dist_cache

        cache = self._get_cache()
        meta_data = cache["meta_data"]
        distributions = np.copy(cache["distributions"])

        if always_minimise:
            # Always minimise
            if len(meta_data["objectives"]) > 1 and isinstance(self.minimise, dict):
                for oid, o in enumerate(meta_data["objectives"]):
                    logging.debug("flip obj")
                    distributions[:, :, oid] *= 1 if self.minimise[o] else -1  # flip
            elif not self.minimise:
                logging.debug("flip all")
                distributions *= -1  # flip

        if lock:
            self._lock_dist = True
            self._dist_cache = np.copy(distributions)

        return distributions

    def get_ranking(self) -> pd.DataFrame:
        """
        Generates a robust ranking which groups statistically equal algorithm together.

        Returns:
            Dataframe with the algorithm as index and a columns with the rank (group) and a column with the mean
            performance over all the bootstrap samples
        """
        cache = self._get_cache()
        distributions = self._get_distributions(always_minimise=True, lock=True)
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        groups = {}
        groupid = 0

        replace = np.inf  # if self.minimise else -np.inf
        argfunc = np.argmin  # if self.minimise else np.argmax

        fractional_wins = np.zeros(n_algorithms, dtype=float)
        algos, wins = np.unique(argfunc(distributions, axis=0), return_counts=True)
        fractional_wins[algos] = wins
        fractional_wins = fractional_wins / self.bootstrap_runs

        candidates_mask = np.ones(n_algorithms, dtype=bool)
        while np.count_nonzero(candidates_mask) > 0:
            logging.info(f"Round {groupid}")
            # Find the winner amongst the remaining candidates
            active_candidates = np.argwhere(candidates_mask).flatten()
            distwins = np.argmin(distributions[candidates_mask, :], axis=0)
            algos, wins = np.unique(distwins, return_counts=True)

            most_wins = np.max(wins)
            winners = np.where(wins == most_wins)[0]
            winner = active_candidates[algos[np.random.choice(winners)]]

            # distwins = np.argmin(distributions, axis=0)
            # algos, wins = np.unique(distwins, return_counts=True)
            # winner = algos[np.argmax(wins)]

            logging.info(
                f"> {meta_data['algorithms'][winner]} has with {np.max(wins)/self.bootstrap_runs:.3%} the most "
                f"wins out of all {np.count_nonzero(candidates_mask)} candidates.")
            groups[groupid] = [(winner, np.mean(distributions[winner, :]))]
            candidates_mask[winner] = False  # Remove winner from candidates
            # distributions[winner, :] = replace
            if np.count_nonzero(candidates_mask) == 0:
                break

            # Perform statistical tests to find candidates that are statistically tied
            candidates = np.argwhere(candidates_mask).flatten()
            pvalues = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                pvalues[i] = self._statistical_test(
                    winner, candidate)  # H0: winner >= candidate: winner is worse or equal than candidate
                logging.info(f"\t> {meta_data['algorithms'][winner]} loses "
                             f"from {meta_data['algorithms'][candidate]} {pvalues[i]:.3%} times.")
            # Multiple test correction
            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            pvalues_order = np.argsort(pvalues)
            # print(f"Round: p-values {pvalues}")
            # reject = pvalues < self.alpha  # no correction
            # reject = multipletests(pvalues, self.alpha, method="holm")[0]  # hommel

            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            # Holm-Bonferroni
            reject = np.zeros(len(candidates), dtype=bool)  # Do not reject any test by default
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
                logging.info(f"\t> {meta_data['algorithms'][candidate]} is with tied with the winner.")
                groups[groupid].append((candidate, 0))
                candidates_mask[candidate] = False
                distributions[candidate, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            # group wins
            dist = copy.copy(cache["distributions"])
            group_wins = np.zeros(len(algorithms), dtype=float)
            logging.info([a[0] for a in algorithms])
            algos, wins = np.unique(argfunc(dist[[a[0] for a in algorithms], :], axis=0), return_counts=True)
            group_wins[algos] = wins
            group_wins = group_wins / self.bootstrap_runs
            algmap = {a: i for i, a in enumerate([a[0] for a in algorithms])}

            for (algorithm, performance) in algorithms:
                results.append({
                    "algorithm": meta_data["algorithms"][algorithm],
                    "group": group + 1,
                    "ranked 1st": fractional_wins[algorithm],
                    "group wins": group_wins[algmap[algorithm]],
                    "ci_mean": np.mean(dist[algorithm, :]),
                    "ci_median": np.median(dist[algorithm, :]),
                    "ci_lb": np.quantile(dist[algorithm, :], self.alpha / 2),
                    "ci_ub": np.quantile(dist[algorithm, :], 1 - self.alpha / 2),
                })

        df = pd.DataFrame(results).set_index("algorithm").sort_values(["group", "ranked 1st", "ci_mean"],
                                                                      ascending=[True, False, self.minimise])
        df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        self._unlock_distribution()

        return df

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Creates a  table of all comparisons between all the algorithms in the benchmark

        Returns:
            dataframe
        """
        cache = self._get_cache()
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        rows = []
        for s1, s2 in itertools.product(range(n_algorithms), repeat=2):
            rows.append({
                "s1": meta_data["algorithms"][s1],
                "s2": meta_data["algorithms"][s2],
                "wins": self._statistical_test(s1, s2) * self.bootstrap_runs
            })

        return pd.DataFrame(rows).set_index(["s1", "s2"]).unstack("s2")

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
        alldf = []
        for obj_id, obj in enumerate(meta_data["objectives"]):
            df = pd.DataFrame(confidence_bounds[:, :, obj_id].T, columns=["median", "lb", "ub"])
            df["algorithm"] = meta_data["algorithms"]
            # if len(meta_data["objectives"]) > 1:
            df["objective"] = obj
            df = df.set_index(["algorithm", "objective"])
            # else:
            #     df = df.set_index(["algorithm"])
            alldf.append(df)

        return pd.concat(alldf)

    def get_bootstrap_distribution(self, algorithm: str):
        """Helper function to get the boostrap distribution for a given algorithm."""
        cache = self._get_cache()
        if algorithm not in cache["meta_data"]["algorithms"]:
            raise LookupError(f"No distribution for '{algorithm}' found!")
        index = cache["meta_data"]["algorithms"].index(algorithm)
        return cache["distributions"][index, :]

    def compute_instance_importance(self, seed: int = 42, resolution=256) -> pd.DataFrame:
        """
        Compute the effect of each instance on the overall performance.

        TODO (very) experimental and probably not working.
        """
        cache = self._get_cache()
        benchmark = self.benchmark
        instances = cache["meta_data"]["instances"]
        minimum = np.min(cache["distributions"])
        maximum = np.max(cache["distributions"])

        class KDE(object):
            """Kernel density estimation helper class"""

            def __init__(self, res, lb, ub):
                """Initialize constants."""
                self.res = res
                self.lb = lb
                self.ub = ub

            def __call__(self, array):
                """Make class callable"""
                x = np.linspace(self.lb, self.ub, self.res)
                kde = gaussian_kde(array)
                return kde(x)

        kde_method = KDE(resolution, minimum, maximum)
        default_kdes = np.apply_along_axis(kde_method, 1, cache["distributions"])

        effects = np.zeros(len(instances))
        for i, instance in enumerate(instances):
            print(instance)
            temp = copy.copy(instances)
            del temp[i]
            self.benchmark = benchmark.filter(instances=temp)
            self.compute()
            cache = self._get_cache()
            distributions = cache["distributions"]

            kdes = np.apply_along_axis(kde_method, 1, distributions)
            kdes = np.abs(kdes - default_kdes)
            kdes = np.sum(kdes)  # TODO check how to aggregate over the algorithms
            effects[i] = kdes

        # TODO find threshold for significant effect of an instance on distributions
        # Normalize
        # effects = 100*(effects / np.sum(effects))

        df = pd.DataFrame(effects, columns=["effect"])
        df["instance"] = instances
        df = df.set_index("instance")

        # Restore
        self.benchmark = benchmark
        self.cache = cache

        return df.sort_values("effect", ascending=False)
