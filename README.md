# Robust Ranking

!Warning! This package is currently still under heavy development. 

A Python package for robust ranking. 

## Installation

```
pip install git+https://github.com/jeroenrook/robustranking
```

## Example usage

```
from robustranking.benchmark import Benchmark
from robustranking.comparison import BootstrapComparison

# Load benchmark data
benchmark = Benchmark()
benchmark.from_pandas(df, "algorithm", "instance", "pqr10")
print(benchmark.show_stats())

comparison = BootstrapComparison(benchmark,
                                 alpha=0.05,
                                 minimise=True,
                                 bootstrap_runs=10000,
                                 aggregation_method=np.mean)

robust_ranks = comparison.get_ranking()

print(robust_ranks)

```

