import numpy as np
import numba as nb


@nb.njit(cache=True)
def _mix_u64(x):
    # x is uint64
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


@nb.njit(cache=True)
def _ceil_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@nb.njit(cache=True)
def build_int_set_same_dtype(keys, empty_sentinel):
    """Builds a new int hash set.

    Args:
        keys: Any integer dtype
        empty_sentinel: Same dtype, typically dtype max.
    """
    cap = _ceil_pow2(keys.size * 2 + 1)
    table = np.empty(cap, dtype=keys.dtype)
    table[:] = empty_sentinel
    mask = cap - 1

    for k in keys:
        # IMPORTANT: sentinel must not appear in keys.
        # Hash in uint64 space regardless of dtype:
        h = _mix_u64(np.uint64(k))
        i = int(h & np.uint64(mask))
        while True:
            cur = table[np.uint64(i)]
            if cur == empty_sentinel or cur == k:
                table[np.uint64(i)] = k
                break
            i = np.uint64((i + 1) & mask)

    return table, mask


@nb.njit(cache=True)
def int_set_contains_same_dtype(table, mask, key, empty_sentinel):
    h = _mix_u64(np.uint64(key))
    i = np.uint(h & np.uint64(mask))
    while True:
        cur = int(table[np.uint64(i)])
        if cur == empty_sentinel:
            return False
        if cur == key:
            return True
        i = np.uint64((i + 1) & mask)
