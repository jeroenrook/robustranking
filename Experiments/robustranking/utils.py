from abc import ABC
import copy
import numpy as np



class PAR(ABC):
    """
    Custom aggregation function for 1-d arrays
    """
    def __init__(self, k=10, cutoff=60):
        """

        Args:
            k: the penalty factor
            cutoff: the cutoff time to be considered as timeout
        """
        self.k = k
        self.cutoff = cutoff

    def __call__(self, array):
        """

        Args:
            array: numpy array
        Returns:
            PARk
        """
        array = copy.copy(array)
        array[array >= self.cutoff] = self.k * self.cutoff
        return np.mean(array)
