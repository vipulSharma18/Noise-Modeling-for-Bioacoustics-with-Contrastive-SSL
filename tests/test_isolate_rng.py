"""
Functional testing of isolate_rng which helps with deterministic corruption for datasets.
"""
from ssl_bioacoustics.utils import isolate_rng
from stable_ssl.utils import seed_everything
import numpy as np


def test_local_state(global_seed=0, local_seed=1):
    """
    Test that the local state works & is isolated.
    1. We need to get the same result within 2 calls with same local seed.
    2. The local state should not "leak" or affect the global state, i.e.,
    the global state should not be the continuation of the local state.
    """
    seed_everything(global_seed)

    # focusing on np ops because at least the CV corruptions just use np
    state_1 = np.random.random()
    with isolate_rng(local_seed):
        first_local_result = np.random.random()
    state_2 = np.random.random()
    with isolate_rng(local_seed):
        second_local_result = np.random.random()
    state_3 = np.random.random()

    # tests that within the local context, we get the same result with the same seed.
    assert first_local_result == second_local_result
    # tests that the global state after state_1 op was different than before it, i.e., BAU.
    assert state_1 != state_2
    # test that the global state after leaving the context is not always the same, i.e., it's not a continuation of the local state.
    assert state_2 != state_3


def test_global_state(global_seed=0, local_seed=1):
    """
    This tests that the global state returned after isolate_rng context is the
    same as the state achieved without any context, i.e., the context is isolated.
    """
    seed_everything(global_seed)
    state_1 = np.random.random()
    state_2 = np.random.random()

    seed_everything(global_seed)
    state_3 = np.random.random()
    with isolate_rng(local_seed):
        state_4 = np.random.random()
    state_5 = np.random.random()

    assert state_1 == state_3  # tests seed_everything resets the global state
    assert state_2 == state_5  # tests context doesn't influence global state
    assert state_2 != state_4  # tests context actually changes the state


def test_diff_local_states(global_seed=0, local_seed1=1, local_seed2=2):
    """
    Test that the context is sensitive to the local seed.
    """
    seed_everything(global_seed)
    state_1 = np.random.random()
    with isolate_rng(local_seed1):
        state_2 = np.random.random()
    with isolate_rng(local_seed2):
        state_3 = np.random.random()

    assert state_1 != state_2
    assert state_1 != state_3
    assert state_2 != state_3  # tests that the local state is sensitive to the local seed
