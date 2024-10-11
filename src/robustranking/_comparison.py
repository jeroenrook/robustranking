import copy
import itertools
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygmo import fast_non_dominated_sorting
from scipy.stats import gaussian_kde
from statsmodels.stats.multitest import multipletests
from typing_extensions import Self

# Local imports
from robustranking.benchmark import Benchmark


class AbstractAlgorithmComparison(ABC):
    """
        Abstract class for algorithm comparison
    """

    def __init__(self, benchmark: Benchmark = Benchmark(), minimise: bool = True):
        """
        Args:
            benchmark: Benchmark class
            minimise: Whether it is the goal to minimise or maximise the objectives
        """
        if not benchmark.check_complete():
            warnings.warn("Benchmark is not complete!")
        self.benchmark = benchmark
        self.minimise = minimise

        self._cache = None

    def load(self, benchmark: Benchmark):
        """
        Load a benchmark class and performs checks
        Args:
            benchmark:

        Returns:

        """
        if not self.benchmark.check_complete():
            warnings.warn("Benchmark is not complete!")

        self.benchmark = benchmark

    def _get_cache(self):
        """
        Cache function is used to store compute intensive intermediate results
        Returns:

        """
        if self._cache is None:
            warnings.warn("No results found. Start computations")
            self.compute()
        return self._cache

    @abstractmethod
    def compute(self) -> Self:
        """
        Abstract method used to compute the intermediate results which are stored in the cache
        Returns:

        """
        raise NotImplementedError("Abstract algorithm comparison class has no comparison functionality")

    @abstractmethod
    def get_ranking(self) -> pd.DataFrame:
        """
        Abstract method for computing the ranking
        Returns:

        """
        raise NotImplementedError("Abstract algorithm comparison class has no ranking functionality")


class AggregatedComparison(AbstractAlgorithmComparison):

    def __init__(
            self,
            benchmark: Benchmark = Benchmark(),
            minimise: [bool | dict] = True,
            aggregation_method=np.mean,
    ):
        super().__init__(benchmark, minimise)

        self.aggregation_method = aggregation_method

    def compute(self) -> Self:
        assert self.benchmark.check_complete()
        array, meta_data = self.benchmark.to_numpy()

        if len(meta_data["objectives"]) > 1 and isinstance(self.aggregation_method, dict):
            aggregation = np.zeros(shape=(len(meta_data["algorithms"]), len(meta_data["objectives"])))
            for o in meta_data["objectives"]:
                agg = self.aggregation_method[o]
                objindex = meta_data["objectives"].index(o)
                aggregation[:, objindex] = np.apply_along_axis(agg, 1, array[:, :, objindex])
        else:
            aggregation = np.apply_along_axis(self.aggregation_method, 1, array)

        self._cache = {"array": array, "meta_data": meta_data, "aggregation": aggregation}

        return self

    def get_ranking(self) -> pd.DataFrame:
        cache = self._get_cache()
        meta_data = cache["meta_data"]

        aggregation = cache["aggregation"]
        if len(meta_data["objectives"]) > 1 and isinstance(self.aggregation_method, dict):
            direction = [1 if self.minimise[o] else -1 for o in meta_data["objectives"]]
        else:
            direction = 1 if self.minimise else -1
        ranks = np.argsort(direction * aggregation, axis=0)
        ranks = np.argsort(ranks, axis=0)  # Sort the ranks to make a mapping to the indexing
        ranks = ranks + 1

        results = []
        for i, algorithm in enumerate(meta_data["algorithms"]):
            result = {
                "algorithm": algorithm,
            }
            if len(meta_data["objectives"]) == 1:
                result["score"] = aggregation[i, 0]
                result["rank"] = ranks[i, 0]
            else:
                for j, objective in enumerate(meta_data["objectives"]):
                    result[(objective, "rank")] = ranks[i, j]
                    result[(objective, "score")] = aggregation[i, j]

            results.append(result)

        df = pd.DataFrame(results).set_index("algorithm")
        if len(meta_data["objectives"]) > 1:
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=[None, "objective"])

        return df


class BootstrapComparison(AbstractAlgorithmComparison):

    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: bool = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int | np.random.RandomState] = 42):
        """

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

    def _get_samples(self, num_instances: int, bootstrap_runs: int | None = None) -> np.ndarray:
        """
        Generates the samples
        Args:
            array:
            meta_data:

        Returns:

        """
        bootstrap_runs = self.bootstrap_runs if bootstrap_runs is None else bootstrap_runs

        # TODO add check to see if there is a sufficient chance of obtaining unique samples.
        #   With a small number of instances this can happen.

        return self.rng.choice(
            np.arange(0, num_instances),
            size=(num_instances, bootstrap_runs),
            replace=True,
        )

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
        distributions = cache["distributions"]

        if self.minimise:
            wins = np.count_nonzero(distributions[s1, :] >= distributions[s2, :])
        else:
            wins = np.count_nonzero(distributions[s1, :] <= distributions[s2, :])

        p_value = wins / self.bootstrap_runs  # p-value

        # p_value < self.alpha -> reject -> s1 is better performing than s2
        # p_value >= self.alpha -> accept -> s1 is equal or worse performing than s2
        return p_value

    def statistical_test(self, algorithm1, algorithm2) -> float:
        cache = self._get_cache()
        algorithms = cache["meta_data"]["algorithms"]
        s1 = algorithms.index(algorithm1)
        s2 = algorithms.index(algorithm2)
        return self._statistical_test(s1, s2)

    def compute(self) -> Self:
        """
        Computes the bootstrap samples and collects for each algorithm the aggregated performance from each bootstrap
        sample.
        Returns:

        """
        assert self.benchmark.check_complete(), "Benchmark table is not complete. Cannot compare."
        array, meta_data = self.benchmark.to_numpy()

        if len(array.shape) == 2:
            array = array.reshape(-1, -1, 1)
        if len(array.shape) != 3:
            raise "Dimension error"

        # Generate n bootstrap samples from the instances
        bootstraps = self._get_samples(len(meta_data["instances"]))
        print(f"{bootstraps.shape=}")

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

    def get_ranking(self) -> pd.DataFrame:
        """
        Generates a robust ranking which groups statistically equal algorithm together.
        Returns:
            Dataframe with the algorithm as index and a columns with the rank (group) and a column with the mean
            performance over all the bootstrap samples
        """
        cache = self._get_cache()
        distributions = copy.copy(cache["distributions"])
        meta_data = cache["meta_data"]
        n_algorithms = len(meta_data["algorithms"])

        groups = {}
        groupid = 0

        replace = np.inf if self.minimise else -np.inf
        argfunc = np.argmin if self.minimise else np.argmax

        fractional_wins = np.zeros(n_algorithms, dtype=float)
        algos, wins = np.unique(argfunc(distributions, axis=0), return_counts=True)
        fractional_wins[algos] = wins
        fractional_wins = fractional_wins / self.bootstrap_runs

        candidates_mask = np.ones(n_algorithms, dtype=bool)
        while np.count_nonzero(candidates_mask) > 0:
            print(f"Round {groupid}")
            # Find the winner amongst the remaining candidates
            distwins = argfunc(distributions, axis=0)
            algos, wins = np.unique(distwins, return_counts=True)
            winner = algos[np.argmax(wins)]
            print(
                f"> {meta_data['algorithms'][winner]} has with {np.max(wins)/self.bootstrap_runs:.3%} the most wins out of all {np.count_nonzero(candidates_mask)} candidates."
            )
            groups[groupid] = [(winner, np.mean(distributions[winner, :]))]
            candidates_mask[winner] = False  # Remove winner from candidates
            distributions[winner, :] = replace
            if np.count_nonzero(candidates_mask) == 0:
                break

            # Perform statistical tests to find candidates that are statistically tied
            candidates = np.argwhere(candidates_mask).flatten()
            pvalues = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                pvalues[i] = self._statistical_test(winner, candidate)  # H0: winner <= candidate
                print(
                    f"\t> {meta_data['algorithms'][winner]} loses from {meta_data['algorithms'][candidate]} {pvalues[i]:.3%} times."
                )
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
                print(f"\t> {meta_data['algorithms'][candidate]} is with tied with the winner.")
                groups[groupid].append((candidate, 0))
                candidates_mask[candidate] = False
                distributions[candidate, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            # group wins
            dist = copy.copy(cache["distributions"])
            group_wins = np.zeros(len(algorithms), dtype=float)
            print([a[0] for a in algorithms])
            algos, wins = np.unique(argfunc(dist[[a[0] for a in algorithms], :], axis=0), return_counts=True)
            group_wins[algos] = wins
            group_wins = group_wins / self.bootstrap_runs
            algmap = {a: i for i, a in enumerate([a[0] for a in algorithms])}

            for (algorithm, performance) in algorithms:
                results.append({
                    "algorithm": meta_data["algorithms"][algorithm],
                    "group": group + 1,
                    "ranked 1st": fractional_wins[algorithm],
                    "group wins": group_wins[algmap[algorithm]]
                })

        df = pd.DataFrame(results).set_index("algorithm").sort_values(["group", "ranked 1st"], ascending=[True, False])
        df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        return df

    def get_comparison_table(self) -> pd.DataFrame:
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
        distributions = cache["distributions"]
        meta_data = cache["meta_data"]

        alpha = self.alpha if alpha is None else alpha
        lower_bound = alpha / 2
        median = 0.5
        upper_bound = 1 - (alpha / 2)

        confidence_bounds = np.quantile(distributions, (median, lower_bound, upper_bound), axis=1)
        alldf = []
        for obj_id, obj in enumerate(meta_data["objectives"]):
            df = pd.DataFrame(confidence_bounds[:, :, obj_id].T, columns=["median", "lb", "ub"])
            df["objective"] = obj
            df["algorithm"] = meta_data["algorithms"]
            df = df.set_index(["algorithm", "objective"])
            alldf.append(df)

        return pd.concat(alldf)

    def get_bootstrap_distribution(self, algorithm: str):
        cache = self._get_cache()
        if algorithm not in cache["meta_data"]["algorithms"]:
            raise LookupError(f"No distribution for '{algorithm}' found!")
        index = cache["meta_data"]["algorithms"].index(algorithm)
        return cache["distributions"][index, :]

    def compute_instance_importance(self, seed: int = 42, resolution=256) -> pd.DataFrame:
        cache = self._get_cache()
        benchmark = self.benchmark
        instances = cache["meta_data"]["instances"]
        minimum = np.min(cache["distributions"])
        maximum = np.max(cache["distributions"])

        class KDE(object):
            """
            Kernel density estimation
            """

            def __init__(self, res, lb, ub):
                self.res = res
                self.lb = lb
                self.ub = ub

            def __call__(self, array):
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
        super().__init__(benchmark, minimise, bootstrap_runs, alpha, aggregation_method, rng)

    @staticmethod
    def _dominates(x, y):
        """
            assumes minimising.
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

        # TODO dominance

        # TODO handle non-numeric objectives
        if isinstance(self.minimise, dict):
            for i, obj in enumerate(cache["meta_data"]["objectives"]):
                if self.minimise[obj] is False:
                    dist[:, :, i] = -1 * dist[:, :, i]
        elif self.minimise is False:
            dist = -1 * dist
        # TODO how to handle uncomparable situations?
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
                pvalues[i] = self._statistical_test(winner, candidate)  # H0: winner <= candidate
            # Multiple test correction
            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            pvalues_order = np.argsort(pvalues)
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
                groups[groupid].append((candidate, 0))
                candidates_mask[candidate] = False
                dist[candidate, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            # group wins
            group_wins = np.zeros(len(algorithms), dtype=float)
            algos, wins = np.unique(argfunc(dist[algorithms, :], axis=0), return_counts=True)
            group_wins[algos] = wins
            group_wins = group_wins / self.bootstrap_runs
            algmap = {a: i for i, a in enumerate(algorithms)}

            for (algorithm, performance) in algorithms:
                results.append({
                    "algorithm": meta_data["algorithms"][algorithm],
                    "group": group + 1,
                    "ranked 1st": fractional_wins[algorithm],
                    "group wins": group_wins[algmap[algorithm]]
                })

        df = pd.DataFrame(results).set_index("algorithm").sort_values(["group", "ranked 1st"], ascending=[True, False])
        df["remaining"] = (1 - df["ranked 1st"].cumsum()).round(4)

        return df


class SubSetComparison(BootstrapComparison):

    def __init__(
        self,
        *args,
        subset_size: int = 2,
        **kwargs,
    ):
        """

        Args:
            benchmark: Benchmark class
            minimise: Whether it is the goal to minimise or maximise the objectives
            subset_size: the subset permutations
            alpha: the alpha value
            aggregation_method: a 1-D aggregation function. Default is np.mean
            rng: Random number generator.
        """
        super().__init__(*args, **kwargs)

        self.subset_size = subset_size

    def _get_samples(self, array: np.ndarray, meta_data: dict) -> np.ndarray:
        subsets = list(itertools.combinations(range(len(meta_data["instances"])), r=self.subset_size))
        self.bootstrap_runs = len(subsets)
        print(f"{self.bootstrap_runs} Samples")
        bootstraps = np.zeros((len(meta_data["instances"]) - self.subset_size, self.bootstrap_runs), dtype=np.int)
        indices = np.arange(0, len(meta_data["instances"]))
        for i, subset in enumerate(subsets):
            bootstraps[:, i] = np.setdiff1d(indices, subset)
        return bootstraps


class AllPermutationsComparison(BootstrapComparison):

    def _get_samples(self, num_instances: int, bootstrap_runs: int | None = None) -> np.ndarray:
        permutations = list(itertools.product(range(num_instances), repeat=num_instances))
        self.bootstrap_runs = len(permutations)
        print(f"{self.bootstrap_runs} Samples")

        bootstraps = np.array(permutations, dtype=int).T
        return bootstraps
