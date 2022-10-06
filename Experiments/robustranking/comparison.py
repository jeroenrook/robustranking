import itertools
from abc import abstractmethod, ABC
import copy
import warnings
import numpy as np
import pandas as pd

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
    def compute(self) -> list:
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


class BootstrapComparison(AbstractAlgorithmComparison):

    def __init__(self,
                 benchmark: Benchmark = Benchmark(),
                 minimise: bool = True,
                 bootstrap_runs: int = 10000,
                 alpha=0.05,
                 aggregation_method=np.mean,
                 rng: [int, np.random.RandomState] = 42):
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

    def _get_samples(self, array: np.ndarray, meta_data: dict) -> np.ndarray:
        """
        Generates the samples
        Args:
            array:
            meta_data:

        Returns:

        """
        return self.rng.choice(np.arange(0, len(meta_data["instances"])),
                               (len(meta_data["instances"]), self.bootstrap_runs),
                               replace=True,)

    def _statistical_test(self, s1: int, s2: int) -> (bool, float):
        """
        Performs a statistical test on the null hypothesis that algorithm 1 (s1) is equal or worse that algorithm 2 (s2)
        Args:
            s1:
            s2:

        Returns:
            p-value of the test. If this value is below the alhpa value, then the hypothesis is rejected and s1 is
            better than s2.
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

    def statistical_test(self, algorithm1, algorithm2):
        cache = self._get_cache()
        algorithms = cache["meta_data"]["algorithms"]
        s1 = algorithms.index(algorithm1)
        s2 = algorithms.index(algorithm2)

        return self._statistical_test(s1, s2)

    def compute(self):
        """
        Computes the bootstrap samples and collects for each algorithm the aggregated performance from each bootstrap
        sample.
        Returns:

        """
        assert self.benchmark.check_complete(), "Benchmark table is not complete. Cannot compare."
        array, meta_data = self.benchmark.to_numpy()

        if len(meta_data["objectives"]) > 1:
            warnings.warn("Benchmark has more than one objective. Using first objective. "
                          "Consider filtering the benchmark. ")
            array = array[:, :, 0]

        # Generate n bootstrap samples from the instances
        # TODO check if samples need to be aligned between the algorithms or not.
        bootstraps = self._get_samples(array, meta_data)

        # Compute the performance of the algorithm on each bootstrap sample
        distributions = np.zeros((len(meta_data["algorithms"]), self.bootstrap_runs))
        for i in range(array.shape[0]):  # Each algorithm
            performance = array[i, :]
            samples = np.take(performance, bootstraps)
            # Compute aggregated performance
            distributions[i, :] = np.apply_along_axis(self.aggregation_method, 0, samples)

        self._cache = {
            "bootstraps": bootstraps,
            "distributions": distributions,
            "meta_data": meta_data,
        }

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

        candidates_mask = np.ones(len(meta_data["algorithms"]), dtype=bool)

        groups = {}
        groupid = 0

        replace = np.inf if self.minimise else -np.inf
        argfunc = np.argmin if self.minimise else np.argmax

        while np.count_nonzero(candidates_mask) > 0:
            # Find the winner amongst the remaining candidates
            distwins = argfunc(distributions, axis=0)
            algos, wins = np.unique(distwins, return_counts=True)
            winner = algos[np.argmax(wins)]
            groups[groupid] = [(winner, np.mean(distributions[winner, :]))]
            candidates_mask[winner] = False  # Remove winner from candidates
            distributions[winner, :] = replace

            # Perform statistical tests to find candidates that are statistically tied
            candidates = np.argwhere(candidates_mask).flatten()
            pvalues = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                pvalues[i] = self._statistical_test(winner, candidate)  # H0: winner <= candidate

            # Multiple test correction
            # TODO iterative method instead of cutoff method as described in paper. Paragraph is illogical
            pvalues_order = np.argsort(pvalues)
            reject = np.zeros(len(candidates), dtype=bool)  # Do not reject any test by default
            for i, index in enumerate(pvalues_order):
                if pvalues[index] < self.alpha / (len(candidates)-i):
                    # Reject H0 -> winner > candidate
                    reject[index] = True
                else:
                    break

            #Not rejecting means they are statistically tied
            ties = candidates[~reject]
            for candidate in ties:
                groups[groupid].append((candidate, np.mean(distributions[candidate, :])))
                candidates_mask[candidate] = False
                distributions[candidate, :] = replace

            groupid += 1

        results = []
        for group, algorithms in groups.items():
            for (algorithm, performance) in algorithms:
                results.append({"algorithm": meta_data["algorithms"][algorithm],
                                "rank": group+1,
                                "mean": performance})

        return pd.DataFrame(results).set_index("algorithm").sort_values(["rank", "mean"])

    def get_confidence_intervals(self):
        """
        Computes the upper and lower bounds of the 1-alpha confidence interval.
        Returns:
            A pandas DataFrame with the bounds and the mean performance
        """
        cache = self._get_cache()
        distributions = cache["distributions"]
        meta_data = cache["meta_data"]

        lower_bound = self.alpha / 2
        median = 0.5
        upper_bound = 1 - (self.alpha / 2)

        confidence_bounds = np.quantile(distributions, (median, lower_bound, upper_bound), axis=1)

        df = pd.DataFrame(confidence_bounds.T, columns=["median", "lb", "ub"])
        df["algorithm"] = meta_data["algorithms"]
        df = df.set_index("algorithm")

        return df

    def get_bootstrap_distribution(self, algorithm: str):
        cache = self._get_cache()
        if algorithm not in cache["meta_data"]["algorithms"]:
            raise LookupError(f"No distribution for '{algorithm}' found!")
        index = cache["meta_data"]["algorithms"].index(algorithm)
        return cache["distributions"][index, :]


class SubSetComparison(BootstrapComparison):
    def __init__(self,
                 *args,
                 subset_size: int = 2,
                 **kwargs,):
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
        bootstraps = np.zeros(
            (len(meta_data["instances"]) - self.subset_size,
             self.bootstrap_runs),
            dtype=np.int
        )
        indices = np.arange(0, len(meta_data["instances"]))
        for i, subset in enumerate(subsets):
            bootstraps[:, i] = np.setdiff1d(indices, subset)
        return bootstraps
