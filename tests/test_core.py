# Unit tests for the core functions of the pvsamlab library.

from pvsamlab.core import run_simulation

def test_run_simulation():
    assert callable(run_simulation)
