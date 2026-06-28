import pytest
import numpy as np

from packed_data_structures.transaction_context import (
    RemapOracle,
    oracle_resolve,
    oracle_resolve_array,
)


def test_remap_oracle_resolve():
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
    assert oracle_resolve(0, oracle) == 0
    assert oracle_resolve(1, oracle) == 1

    # Deletions -> missing
    assert oracle_resolve(2, oracle) == 999
    assert oracle_resolve(5, oracle) == 999

    # Moves
    assert oracle_resolve(8, oracle) == 2
    assert oracle_resolve(9, oracle) == 5

    # Staged resolving to addition_destinations
    assert oracle_resolve(10, oracle) == 10
    assert oracle_resolve(11, oracle) == 11

    # Truncation out of bounds
    assert oracle_resolve(15, oracle) == 999


def test_remap_oracle_resolve_array():
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

    resolved = oracle_resolve_array(arr, oracle, inplace=False)

    expected = np.array([0, 1, 999, 3, 2, 999, 999], dtype=np.uint32)
    assert np.array_equal(resolved, expected)
