import numpy as np
import pytest

from final_proj.environment.map_loader import MapLoader
from final_proj.paths import DEFAULT_MAP


@pytest.fixture(scope="session")
def house_grid():
    return MapLoader(DEFAULT_MAP).load()


@pytest.fixture
def open_grid():
    """10x10 fully free grid."""
    return np.zeros((10, 10), dtype=int)
