import numpy as np
import pandas as pd
from typing_extensions import Self

from robustranking.benchmark import Benchmark
# Local imports
from robustranking.comparison.abstract_comparison import \
    AbstractAlgorithmComparison


class AggregatedComparison(AbstractAlgorithmComparison):
    """Vanilla comparison of algorithms based on their performance over all instances."""

    def __init__(
            self,
            benchmark: Benchmark = Benchmark(),
            minimise: bool | dict = True,
            aggregation_method=np.mean,
    ):
        super().__init__(benchmark, minimise)

        self.aggregation_method = aggregation_method

    def compute(self) -> Self:
        """Compute the overall performance over all instances."""
        if not self.benchmark.check_complete():
            raise ValueError("Benchmark is not complete.")

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
        """Create a ranking in a dataframe."""
        cache = self._get_cache()
        meta_data = cache["meta_data"]

        aggregation = cache["aggregation"]
        if len(meta_data["objectives"]) > 1 and isinstance(self.minimise, dict):
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
