import numpy as np
import pandas as pd
from abc import ABC
import copy

# Custom aggregation metrics
class PAR(ABC):
    """
    Custom aggregation function for 1-d arrays
    """
    def __init__(self, cutoff: int, k: int = 10):
        """

        Args:
            k: the penalty factor
            cutoff: the cutoff time to be considered as timeout
        """
        self.cutoff = cutoff
        self.k = k

    def __str__(self):
        return f"PAR{self.k}"

    def __call__(self, array: [np.ndarray | pd.Series]) -> float:
        """

        Args:
            array: numpy array
        Returns:
            PARk
        """
        array = copy.copy(array)
        array[array >= self.cutoff] = self.k * self.cutoff
        return np.mean(array)

#TODO PQRk