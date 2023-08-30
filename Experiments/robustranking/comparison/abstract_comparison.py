import warnings
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing_extensions import Self

# Local imports
from robustranking.benchmark import Benchmark


class AbstractAlgorithmComparison(ABC):
    """
        Abstract class for algorithm comparison
    """

    def __init__(self, benchmark: Benchmark = Benchmark(), minimise: bool | dict = True):
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
        raise NotImplementedError(
            "Abstract algorithm comparison class has no comparison functionality")

    @abstractmethod
    def get_ranking(self) -> pd.DataFrame:
        """
        Abstract method for computing the ranking
        Returns:

        """
        raise NotImplementedError(
            "Abstract algorithm comparison class has no ranking functionality")
