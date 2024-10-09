from typing_extensions import Self

import pandas as pd
import numpy as np

from robustranking.benchmark import Benchmark
from robustranking.comparison.abstract_comparison import AbstractAlgorithmComparison


class RankedComparison(AbstractAlgorithmComparison):
    """Compares algorithms based on their relative performance differences by ranking them on individual instances."""

    def __init__(
        self,
        benchmark: Benchmark = Benchmark(),
        minimise: bool | dict = True,
        aggregation_method=np.mean,
        tie_method=np.mean,
    ):
        super().__init__(benchmark, minimise)

        self.aggregation_method = aggregation_method
        self.tie_method = tie_method

    def compute(self) -> Self:
        """Compute the rankings on the instance level."""
        if not self.benchmark.check_complete():
            raise ValueError("Benchmark is not complete.")

        array, meta_data = self.benchmark.to_numpy()
        # 0 - algo, 1 - instance, 2 - objective

        if len(meta_data["objectives"]) > 1 and isinstance(self.minimise, dict):
            directions = [1 if self.minimise[o] else -1 for o in meta_data["objectives"]]
        else:
            directions = [
                1,
            ] if self.minimise else [
                -1,
            ]

        all_ranks = np.zeros_like(array, dtype=float)
        for objindex, direction in zip(range(len(meta_data["objectives"])), directions):
            oarray = direction * array[:, :, objindex]
            ranks = np.argsort(np.argsort(oarray, axis=0), axis=0)
            ranks += 1
            # Repair duplicates and average them
            with_duplicates = np.apply_along_axis(lambda r: len(np.unique(r)) != len(r), 0, oarray)
            for iindex in np.flatnonzero(with_duplicates):
                row = oarray[:, iindex]
                unique_values, unique_counts = np.unique(row, return_counts=True)
                unique_values = [v for v, c in zip(unique_values, unique_counts) if c > 1]
                for value in unique_values:
                    indices = np.where(row == value)[0]
                    ranks[indices, iindex] = self.tie_method(ranks[indices, iindex])
            all_ranks[:, :, objindex] = ranks

        # Do aggregation TODO: merge with aggregation ranking?
        if len(meta_data["objectives"]) > 1 and isinstance(self.aggregation_method, dict):
            aggregation = np.zeros(shape=(len(meta_data["algorithms"]), len(meta_data["objectives"])))
            for o in meta_data["objectives"]:
                agg = self.aggregation_method[o]
                objindex = meta_data["objectives"].index(o)
                aggregation[:, objindex] = np.apply_along_axis(agg, 1, all_ranks[:, :, objindex])
        else:
            aggregation = np.apply_along_axis(self.aggregation_method, 1, all_ranks)

        self._cache = {"array": array, "meta_data": meta_data, "aggregation": aggregation, "ranks": all_ranks}

        return self

    def get_ranking(self) -> pd.DataFrame:
        """Create a ranking in a dataframe."""
        cache = self._get_cache()
        meta_data = cache["meta_data"]

        aggregation = cache["aggregation"]
        ranks = np.argsort(aggregation, axis=0)
        ranks = np.argsort(ranks, axis=0)  # Sort the ranks to make a mapping to the indexing
        ranks = ranks + 1

        n_objectives = len(meta_data["objectives"])

        results = []
        for i, algorithm in enumerate(meta_data["algorithms"]):
            result = {
                "algorithm": algorithm,
            }
            if n_objectives == 1:
                result["score"] = aggregation[i, 0]
                result["rank"] = ranks[i, 0]
            else:
                for j, objective in enumerate(meta_data["objectives"]):
                    result[(objective, "rank")] = ranks[i, j]
                    result[(objective, "score")] = aggregation[i, j]

            results.append(result)

        df = pd.DataFrame(results).set_index("algorithm")
        if len(meta_data["objectives"]) > 1:
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=["objective", None])

        sort_key = (meta_data["objectives"][0], "rank") if n_objectives > 1 else "rank"
        df = df.sort_values(by=sort_key)

        return df
