import itertools
from abc import ABC

import numpy as np
import pandas as pd
from typing_extensions import Self


class Benchmark(ABC):
    """
    Data structure to store benchmark results

    TODO: Add support for named tuples as item keys
    """

    def __init__(self, algorithms: list = None, instances: list = None, objectives: list = None):
        """
        Initialize the Benchmark class.

        Args:
            algorithms: list of algorithm names
            instances: list of instance names
            objectives: list of objectives
        """
        self.algorithms = []
        self.instances = []
        self.objectives = []

        self._algorithm2index = {}  # algorithm: set(run_data_ids)
        self._instance2index = {}  # instance: set(run_data_ids)
        self._objective2index = {}  # objective: set(run_data_ids)

        self._run_data = []  # value
        self._run_data_keys = []  # (algorithm, instance, objective)

        # Parse argument help function
        def parse_argument(argument, call):
            if argument is not None:
                for item in argument:
                    call(item)

        parse_argument(algorithms, self.add_algorithm)
        parse_argument(instances, self.add_instance)
        parse_argument(objectives, self.add_objective)

    # Functional
    def reset(self):
        """Reset the benchmark data structure."""
        self.algorithms = []
        self.instances = []
        self.objectives = []

        self._algorithm2index = {}
        self._instance2index = {}
        self._objective2index = {}

        self._run_data = []
        self._run_data_keys = []

    def check_complete(self) -> bool:
        """
        Checks if for all possible (algorithm, instance, objective) combination there is a value.

        Returns: bool:
        """
        for a, i, o in itertools.product(self.algorithms, self.instances, self.objectives):
            if len(self._get_indices(a, i, o)) == 0:
                return False
        return True

    def show_stats(self) -> pd.Series:
        """Create statistics and return it as a pd.Series."""
        stats = {
            "algorithms": len(self.algorithms),
            "instances": len(self.instances),
            "objectives": len(self.objectives),
            "values": len(self._run_data),
            "complete": self.check_complete()
        }
        return pd.Series(stats)

    def filter(self, *args, **kwargs) -> Self:
        """
        Creates a new benchmark class where the contents are filtered based on the given arguments.

        Args:
            *args: passed along to self.get
            **kwargs: passed along to self.get

        Returns: Benchmark:

        """
        new_benchmark = Benchmark()
        runs = self.get(*args, **kwargs)
        for (a, i, o), v in runs.items():
            new_benchmark.add_run(a, i, o, v)
        return new_benchmark

    # Get functions
    def _get_indices(self,
                     algorithms: [list, str] = None,
                     instances: [list, str] = None,
                     objectives: [list, str] = None) -> set:
        """
        Retrieves the indices of all the values of all combinations of the items in the provided dimensions.

        When there is no filter on one dimension then it considers all available items in that dimension.

        Args:
            algorithms:
            instances:
            objectives:

        Returns:
            A set of indices
        """

        def item_indices(items, item_list, item2index):
            if not isinstance(items, list):
                return item2index[items]

            indices = []
            for item in items:
                assert item in item_list, f"Unknown item {item}"
                indices.append(item2index[item])

            indices = indices[0].union(*indices[1:])
            return indices

        indices = []
        if algorithms is not None:
            indices.append(item_indices(algorithms, self.algorithms, self._algorithm2index))
        if instances is not None:
            indices.append(item_indices(instances, self.instances, self._instance2index))
        if objectives is not None:
            indices.append(item_indices(objectives, self.objectives, self._objective2index))

        if len(indices) == 0:
            indices = list(range(len(self._run_data)))  # return all
        elif len(indices) == 1:
            indices = indices[0]
        else:
            indices = indices[0].intersection(*indices[1:])

        return indices

    def get(self, algorithms=None, instances=None, objectives=None) -> dict:
        """
        Retrieves all the keys and values of all combinations of the items in the provided dimensions.

        When there is no filter on one dimension then it considers all available items in that dimension.

        Args:
            algorithms:
            instances:
            objectives:

        Returns:
            A dictionary with a key a tuple of the algorithm, instance and objective and as value the datapoint
        """
        indices = self._get_indices(algorithms, instances, objectives)

        output = {}
        for index in indices:
            output[self._run_data_keys[index]] = self._run_data[index]
        return output

    # Add
    def add_algorithm(self, name):
        """
        Add an algorithm.

        Args:
            name:
        """
        if name not in self.algorithms:
            self.algorithms.append(name)
            self._algorithm2index[name] = set()

    def add_instance(self, name):
        """
        Add an instance.

        Args:
            name:
        """
        # assert name not in self.instances, "Instance already exists"
        if name not in self.instances:
            self.instances.append(name)
            self._instance2index[name] = set()

    def add_objective(self, name):
        """
        Add an objective.

        Args:
            name:
        """
        # assert name not in self.objectives, "Objective already exists"
        if name not in self.objectives:
            self.objectives.append(name)
            self._objective2index[name] = set()

    def add_run(self, algorithm, instance, objective, value, replace=True):
        """
        Adds a value for the provided key.

        Args:
            algorithm:
            instance:
            objective:
            value: data point
            replace: bool: Whether to replace a value when there already exists one for the given key
        """
        if algorithm not in self.algorithms:
            self.add_algorithm(algorithm)
        if instance not in self.instances:
            self.add_instance(instance)
        if objective not in self.objectives:
            self.add_objective(objective)

        # Prevent duplicates
        indices = self._get_indices(algorithm, instance, objective)
        if len(indices) == 1:
            if replace:
                index = indices.pop()
                self._run_data[index] = value
            else:
                # warnings.WarningMessage("Entry already in table. Ignoring this entry")
                print("Entry already in table. Ignoring this entry")
        else:
            index = len(self._run_data)
            self._run_data.append(value)
            self._run_data_keys.append((algorithm, instance, objective))

            self._algorithm2index[algorithm].add(index)
            self._instance2index[instance].add(index)
            self._objective2index[objective].add(index)

    # Imports
    def from_pandas(self, df, algorithm_key: [str | tuple], instance_key: [str | tuple],
                    objective_keys: [str | tuple | list]):
        """
        Generate the class from a pandas DataFrame.

        Each key component should be a column as well as the value key.
        Other columns are ignored.

        Args:
            df:
            algorithm_key:
            instance_key:
            objective_key:
            value_key:
        """
        self.reset()

        for algorithm in df[algorithm_key].unique():
            self.add_algorithm(algorithm)
        for instance in df[instance_key].unique():
            self.add_instance(instance)
        if not isinstance(objective_keys, list):
            objective_keys = [objective_keys]
        for objective_key in objective_keys:
            self.add_objective(objective_key)

        for key, row in df.iterrows():
            for objective_key in objective_keys:
                self.add_run(row[algorithm_key], row[instance_key], objective_key, row[objective_key])

        return self

    # Exports
    def to_pandas(self, *args, **kwargs) -> pd.DataFrame:
        """
        Creates a pandas DataFrame out of the benchmark class.

        The keys are set as index which makes stacking and unstacking easy to do.

        Passing along filters on the key elements is possible.
        Args:
            *args:
            **kwargs:

        Returns: Dataframe
        """
        items = []
        for k, v in self.get(*args, **kwargs).items():
            item = {"algorithm": k[0], "instance": k[1], "objective": k[2], "value": v}
            items.append(item)
        return pd.DataFrame(items).set_index(["algorithm", "instance", "objective"])

    def to_numpy(self, algorithms=None, instances=None, objectives=None) -> tuple[np.ndarray, dict]:
        """
        Create a 3-dimensional numpy array.

        The dimensions represent the algorithms, instances and objectives, respectively.
        Along with the array, meta-data provided with which algorithms, instance and objective each cell
         represents. Empty cells are filled with np.nan

         Passing along filters on the key elements is possible.

        Args:
            algorithms:
            instances:
            objectives:

        Returns:
            numpy.ndarray, metadata
        """

        def parse_item(items, parent):
            if items is None:
                items = parent
            elif not isinstance(items, list):
                items = [items]
            return items, {pos: item for item, pos in enumerate(items)}

        algorithms, a2index = parse_item(algorithms, self.algorithms)
        instances, i2index = parse_item(instances, self.instances)
        objectives, o2index = parse_item(objectives, self.objectives)

        results = self.get(algorithms=None, instances=None, objectives=None)

        shape = (
            len(algorithms),
            len(instances),
            len(objectives),
        )
        array = np.zeros(shape)
        array.fill(np.nan)

        for (a, i, o), v in results.items():
            array[a2index[a], i2index[i], o2index[o]] = v

        meta_data = {"algorithms": algorithms, "instances": instances, "objectives": objectives}
        return array, meta_data
