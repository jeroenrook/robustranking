import itertools

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from robustranking.benchmark import Benchmark
from robustranking.comparison import (AggregatedComparison, BootstrapComparison, SubSetComparison)
from robustranking.utils import *

#Artificial
# competition = Benchmark()
# algorithms = [f"Algorithm-{i}" for i in range(1, 4)]
# instances = [f"Instance-{i}" for i in range(1, 20)]
# objectives = ["Runtime", "Quality"]
#
# rng = np.random.default_rng(0)
#
# for a, i, o in itertools.product(algorithms, instances, objectives):
#     factor = 0.8 if a == "Algorithm-1" else 1
#     competition.add_run(a, i, o, factor * rng.random())
# competition.add_run("Algorithm-1", "Instance-1", "Runtime", 1.0, replace=True)
# competition = competition.filter(objectives="Runtime")

#SAT2022
df = pd.read_csv("./Rundata/sc2022-detailed-results/main-seq.csv")
del df["verified-result"]
del df["claimed-result"]
del df["hash"]
df = df.set_index(["benchmark"])
df = df.stack().reset_index().rename(columns={"level_1": "algorithm", "benchmark": "instance", 0: "PAR2"})
df["Solved"] = df["PAR2"] < 10000  # Solved instances
df = df.set_index(["algorithm", "instance"]).stack().reset_index().rename(columns={"level_2": "objective", 0: "value"})

competition = Benchmark()
competition.from_pandas(df, "algorithm", "instance", "objective", "value")

print("AggregatedComparison")
comparison = AggregatedComparison(
    competition,
    minimise=True,
    aggregation_method=np.sum,
)

default_df = comparison.compute().get_ranking().sort_values(("PAR2", "rank"))

print("BootstrapComparison")
comparison = BootstrapComparison(competition.filter(objectives="PAR2"),
                                 minimise=True,
                                 bootstrap_runs=10000,
                                 alpha=0.1,
                                 aggregation_method=np.mean,
                                 rng=20221007)
comparison.compute()

df = comparison.get_ranking().join(default_df["PAR2"])
print(df.groupby("group").count())
exit()

#print(comparison.compute_instance_importance())

# print("SubSetComparison")
# comparison = SubSetComparison(competition.filter(objectives="Runtime"),
#                                  minimise=True,
#                                  subset_size=3,
#                                  aggregation_method=np.mean,)
# comparison.compute()
#
# print(comparison.get_ranking())
#
plt.rcParams['figure.dpi'] = 300
plot_distribution(comparison, "Kissat_MAB-HyWalk")

fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
axs[0][0].set_title("Group 1 vs Group 1")
plot_distributions_comparison(comparison, ["Kissat_MAB-HyWalk", "CaDiCaL_DVDL_V1"], ax=axs[0][0])
axs[0][1].set_title("Group 1 vs Group 2")
plot_distributions_comparison(comparison, ["Kissat_MAB-HyWalk", "hCaD_V1-psids"], ax=axs[0][1])
axs[1][0].set_title("Group 1 vs Group 3")
plot_distributions_comparison(comparison, ["Kissat_MAB-HyWalk", "LSTech_kissat"], ax=axs[1][0])
axs[1][1].set_title("Group 1 vs Group 6")
plot_distributions_comparison(comparison, ["Kissat_MAB-HyWalk", "IsaSAT"], ax=axs[1][1])
plt.tight_layout()
plt.show()

plot_ci_list(comparison)
plot_ci_list(comparison, top=10)
