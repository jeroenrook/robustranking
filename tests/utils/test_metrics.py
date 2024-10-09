import numpy as np
import pandas as pd

from robustranking.utils.metrics import PAR


def test_par_initialization():
    """Test initialization of PAR object."""
    par = PAR(cutoff=100, k=10)
    assert par.cutoff == 100
    assert par.k == 10


def test_par_str_representation():
    """Test string representation of PAR object."""
    par = PAR(cutoff=100, k=10)
    assert str(par) == "PAR10"

    par = PAR(cutoff=100, k=5)
    assert str(par) == "PAR5"


def test_par_call_with_numpy_array():
    """Test call functionality with numpy array."""
    par = PAR(cutoff=100, k=10)

    # Case 1: Array with no elements above cutoff
    arr = np.array([50, 70, 90])
    result = par(arr)
    assert np.isclose(result, np.mean(arr))

    # Case 2: Array with elements above cutoff
    arr = np.array([50, 70, 110])
    result = par(arr)
    expected_array = np.array([50, 70, 1000])
    assert np.isclose(result, np.mean(expected_array))


def test_par_call_with_pandas_series():
    """Test call functionality with pandas Series."""
    par = PAR(cutoff=100, k=10)

    # Case 1: Series with no elements above cutoff
    series = pd.Series([50, 70, 90])
    result = par(series)
    assert np.isclose(result, series.mean())

    # Case 2: Series with elements above cutoff
    series = pd.Series([50, 70, 110])
    result = par(series)
    # The value 110 is replaced by 10 * 100 = 1000
    expected_series = pd.Series([50, 70, 1000])
    assert np.isclose(result, expected_series.mean())


def test_par_edge_cases():
    """Test edge cases (e.g., empty arrays, all elements above cutoff)"""
    par = PAR(cutoff=60, k=2)

    # Empty array
    empty_array = np.array([])
    result = par(empty_array)
    assert np.isnan(result)

    # All elements above cutoff
    all_above_cutoff = np.array([110, 120, 130])
    result = par(all_above_cutoff)
    expected_result = 120
    assert np.isclose(result, expected_result)
