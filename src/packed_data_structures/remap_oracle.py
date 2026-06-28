from __future__ import annotations

from typing import NamedTuple

import numba as nb
import numpy as np

# Defensive checks that are mathematically unreachable under standard swap-and-pop,
# but preserved as a toggleable safety net for future architectural changes.
_ENABLE_DEFENSIVE_CHECKS = False

class RemapOracle(NamedTuple):
    """An oracle that answers where an index has moved during a bulk edit.

    Because swap-and-pop edits move row indices, references (like foreign keys)
    need to be remapped to their new physical locations. This oracle provides
    the mapping rules for a specific table during a transaction commit.

    Attributes:
        set_null_unlinks_sorted (np.ndarray): Sorted array of indices that should be set to null.
        deletions_sorted (np.ndarray): Sorted array of deleted row indices.
        moves_from (np.ndarray): Sorted array of original row indices that were relocated.
        moves_to (np.ndarray): The new physical indices corresponding to `moves_from`.
        moves_to_sorted (np.ndarray): A sorted version of `moves_to` for fast collision checks.
        addition_destinations (np.ndarray): Array of indices where new staged rows were placed.
        staged_indices_start (int): The index where staged (newly added) rows begin.
        new_size (int): The total physical size of the table after the edit.
        missing_index_sentinel (int): The integer sentinel representing a missing link.
    """

    set_null_unlinks_sorted: np.ndarray
    deletions_sorted: np.ndarray
    moves_from: np.ndarray
    moves_to: np.ndarray
    moves_to_sorted: np.ndarray
    addition_destinations: np.ndarray
    staged_indices_start: int
    new_size: int
    missing_index_sentinel: int


@nb.njit(inline="always")
def oracle_resolve(idx: int, oracle: RemapOracle) -> int:
    """Translates a single index through a RemapOracle.

    This resolves:
      - staged indices (newly added entries)
      - moved indices (swap-and-pop relocations)
      - overwritten indices
      - explicit deletions

    Args:
        idx (int): The original index to resolve.
        oracle (RemapOracle): The oracle containing mapping rules.

    Returns:
        int: The final physical index, or the missing sentinel if deleted/voided.
    """
    o = oracle
    missing_idx = o.missing_index_sentinel

    # Staged IDs (New Additions)
    if idx >= o.staged_indices_start:
        offset = idx - o.staged_indices_start
        if offset < len(o.addition_destinations):
            return int(o.addition_destinations[offset])
        return missing_idx

    # Explicit Deletions
    if len(o.deletions_sorted) > 0:
        # Optimization: fast range check before binary search
        if o.deletions_sorted[0] <= idx <= o.deletions_sorted[-1]:
            i_del = np.searchsorted(o.deletions_sorted, idx)
            if i_del < len(o.deletions_sorted) and o.deletions_sorted[i_del] == idx:
                return missing_idx

    # Moves (Did I move?)
    if len(o.moves_from) > 0:
        if o.moves_from[0] <= idx <= o.moves_from[-1]:
            i_move = np.searchsorted(o.moves_from, idx)
            if i_move < len(o.moves_from) and o.moves_from[i_move] == idx:
                return int(o.moves_to[i_move])

    if _ENABLE_DEFENSIVE_CHECKS:  # pragma: no cover
        # Overwritten by Move (Did someone move here?)
        if len(o.moves_to_sorted) > 0:
            if o.moves_to_sorted[0] <= idx <= o.moves_to_sorted[-1]:
                i_over = np.searchsorted(o.moves_to_sorted, idx)
                if i_over < len(o.moves_to_sorted) and o.moves_to_sorted[i_over] == idx:
                    return missing_idx

        # Overwritten by Addition (Did an addition land here?)
        if len(o.addition_destinations) > 0:
            if o.addition_destinations[0] <= idx <= o.addition_destinations[-1]:
                i_add = np.searchsorted(o.addition_destinations, idx)
                if (
                    i_add < len(o.addition_destinations)
                    and o.addition_destinations[i_add] == idx
                ):
                    return missing_idx

        # Truncation (Array shrunk)
        if idx >= o.new_size:
            return missing_idx

    return idx


@nb.njit(cache=True, parallel=True)
def oracle_resolve_array(
    values: np.ndarray,
    oracle: RemapOracle,
    inplace: bool = False,
) -> np.ndarray:
    """Translate a list of indices through a RemapOracle.

    Args:
        values (np.ndarray): Array of indices referencing the oracle's table.
        oracle (RemapOracle): RemapOracle for the referenced table.
        inplace (bool): Preform the update in-place on the provided values array.

    Returns:
        np.ndarray: Array with all indices translated to final physical indices.
    """
    if not inplace:
        out = np.zeros_like(values)
    else:
        out = values

    missing = oracle.missing_index_sentinel

    for i in nb.prange(len(values)):
        v = values[i]
        if v == missing:
            out[i] = missing
        else:
            out[i] = oracle_resolve(v, oracle)

    return out
