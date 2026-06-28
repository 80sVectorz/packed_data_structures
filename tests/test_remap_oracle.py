import pytest
import numpy as np

from packed_data_structures.remap_oracle import (
    RemapOracle,
    oracle_resolve,
    oracle_resolve_array,
)

from .testing_utils import get_func_variants


@pytest.mark.parametrize("name, resolve_fn", get_func_variants(oracle_resolve))
def test_remap_oracle_resolve(name, resolve_fn):

    oracle = RemapOracle(
        set_null_unlinks_sorted=np.array([], dtype=np.uint32),
        deletions_sorted=np.array([2, 5], dtype=np.uint32),
        moves_from=np.array([8, 9], dtype=np.uint32),
        moves_to=np.array([2, 5], dtype=np.uint32),
        moves_to_sorted=np.array([2, 5], dtype=np.uint32),
        addition_destinations=np.array([10, 11], dtype=np.uint32),
        staged_indices_start=10,
        new_size=12,
        missing_index_sentinel=999,
    )

    # Unchanged
    assert resolve_fn(0, oracle) == 0
    assert resolve_fn(1, oracle) == 1

    # Deletions -> missing
    assert resolve_fn(2, oracle) == 999
    assert resolve_fn(5, oracle) == 999

    # Moves
    assert resolve_fn(8, oracle) == 2
    assert resolve_fn(9, oracle) == 5

    # Staged resolving to addition_destinations
    assert resolve_fn(10, oracle) == 10
    assert resolve_fn(11, oracle) == 11

    # Truncation out of bounds
    assert resolve_fn(15, oracle) == 999


@pytest.mark.parametrize("name, resolve_array_fn", get_func_variants(oracle_resolve_array))
def test_remap_oracle_resolve_array(name, resolve_array_fn):

    oracle = RemapOracle(
        set_null_unlinks_sorted=np.array([], dtype=np.uint32),
        deletions_sorted=np.array([2], dtype=np.uint32),
        moves_from=np.array([4], dtype=np.uint32),
        moves_to=np.array([2], dtype=np.uint32),
        moves_to_sorted=np.array([2], dtype=np.uint32),
        addition_destinations=np.array([], dtype=np.uint32),
        staged_indices_start=5,
        new_size=4,
        missing_index_sentinel=999,
    )

    arr = np.array([0, 1, 2, 3, 4, 10, 999], dtype=np.uint32)

    resolved = resolve_array_fn(arr, oracle, inplace=False)

    expected = np.array([0, 1, 999, 3, 2, 999, 999], dtype=np.uint32)
    assert np.array_equal(resolved, expected)
