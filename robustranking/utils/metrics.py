import copy
from abc import ABC

import numpy as np
import pandas as pd


# Custom aggregation metrics
class PAR(ABC):
    """Custom aggregation function for 1-d arrays."""

    def __init__(self, cutoff: int, k: int = 10):
        """Initialization function to set constants.

        Args:
            k: the penalty factor
            cutoff: the cutoff time to be considered as timeout
        """
        self.cutoff = cutoff
        self.k = k

    def __str__(self):
        """String representation."""
        return f"PAR{self.k}"

    def __call__(self, array: [np.ndarray | pd.Series]) -> float:
        """Callable function.

        Args:
            array: numpy array
        Returns:
            PARk
        """
        if len(array) == 0:
            return np.nan
        array = copy.copy(array)
        array[array >= self.cutoff] = self.k * self.cutoff
        return np.mean(array)


# TODO PQRk
