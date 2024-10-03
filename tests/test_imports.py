import pytest

def test_import_package():
    try:
        import robustranking
    except ImportError as e:
        pytest.fail(f"Failed to import robustranking: {e}")

def test_import_submodules():
    try:
        import robustranking.comparison
        import robustranking.utils
    except ImportError as e:
        pytest.fail(f"Failed to import submodule: {e}")