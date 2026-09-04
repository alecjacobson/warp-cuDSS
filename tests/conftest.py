import numpy as np
import pytest
import warp as wp

wp.init()


@pytest.fixture(scope="session")
def device():
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA device required for warp_cudss tests")
    return "cuda:0"


@pytest.fixture(autouse=True)
def _seeded_rng():
    return np.random.default_rng(1234)
