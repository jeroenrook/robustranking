import pytest
import os
import tempfile


def test_import_module():
    """Test if normal importing the module works."""
    try:
        import robustranking  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import robustranking: {e}")


def test_import_module_as():
    """Test if importing under a different alias works."""
    try:
        import robustranking as rr  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import robustranking as rr: {e}")


def test_import_submodules():
    """Check if subpackages can be imported."""
    try:
        import robustranking.comparison  # noqa: F401
        import robustranking.utils  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import submodule: {e}")


def test_import_class():
    """Check if importing Benchmark works"""
    try:
        import robustranking  # noqa: F401

        robustranking.Benchmark()
    except ImportError as e:
        pytest.fail(f"Failed to load Benchmark class: {e}")


def test_import_other_dir():
    """Check if import works outside of cwd"""
    tmpdirname = tempfile.mkdtemp()
    os.chdir(tmpdirname)
    try:
        import robustranking  # noqa: F401
        print(os.getcwd())
    except ImportError as e:
        pytest.fail(f"Failed to import robustranking: {e} from {os.getcwd()}")
