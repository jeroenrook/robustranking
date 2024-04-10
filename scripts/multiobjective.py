#!/usr/bin/env python3

import numpy as np
import pandas as pd

from robustranking.benchmark import Benchmark
from robustranking.comparison import BootstrapComparison, MOBootstrapComparison
from robustranking.utils import *

if __name__ == "__main__":
    df = pd.read_csv("./Rundata/sc2022-detailed-results/main-seq.csv")
    del df["verified-result"]
    del df["claimed-result"]
    del df["hash"]
    df = df.set_index(["benchmark"])
    df = df.stack().reset_index().rename(
        columns={"level_1": "algorithm", "benchmark": "instance", 0: "PAR2"})
    df["Solved"] = df["PAR2"] < 10000  # Solved instances
    # df = df.set_index(["algorithm", "instance"])#.stack().reset_index().rename(columns={"level_2": "objective", 0: "value"})

    competition = Benchmark()
    competition.from_pandas(df, "algorithm", "instance", ["PAR2", "Solved"])
    print("Complete table:", competition.check_complete())
    competition.to_pandas().unstack("objective")

    comparison = MOBootstrapComparison(competition,
                                       alpha=0.05,
                                       minimise={"PAR2": True, "Solved": False},
                                       bootstrap_runs=10000,
                                       aggregation_method={"PAR2": np.mean,
                                                           "Solved": np.sum})

    # competition = Benchmark()
    # algorithms = [f"Algorithm-{i}" for i in range(1, 4)]
    # instances = [f"Instance-{i}" for i in range(1, 101)]
    # objectives = ["Runtime", "Quality"]
    #
    # for a, i, o in itertools.product(algorithms, instances, objectives):
    #     factor = 0.8 if a == "Algorithm-1" else 1
    #     competition.add_run(a, i, o, factor * np.random.rand())
    # competition.add_run("Algorithm-1", "Instance-1", "Runtime", 1.0, replace=True)
    #
    # comparison = MOBootstrapComparison(competition,
    #                                    alpha=0.05,
    #                                    minimise=True,
    #                                    bootstrap_runs=10000,
    #                                    aggregation_method=np.mean)

    comparison.compute()

    print(comparison.get_ranking())





