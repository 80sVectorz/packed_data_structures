import pytest
import numpy as np
from packed_data_structures.id_range import IdRange

def test_id_range_basic_properties():
    base_range = range(5, 15, 2)
    idr = IdRange(base_range, np.uint32)
    
    assert idr.start == 5
    assert idr.stop == 15
    assert idr.step == 2
    assert len(idr) == 5
    assert 5 in idr
    assert 6 not in idr

def test_id_range_equality():
    idr1 = IdRange(range(0, 10), np.uint32)
    idr2 = IdRange(range(0, 10), np.uint32)
    idr3 = IdRange(range(0, 10), np.int64)
    
    assert idr1 == idr2
    assert idr1 != idr3
    assert idr1 == range(0, 10)  # Should equate to base range

def test_id_range_iteration():
    idr = IdRange(range(0, 3), np.uint32)
    elements = list(idr)
    
    assert len(elements) == 3
    assert isinstance(elements[0], np.uint32)
    assert elements[0] == 0
    assert elements[2] == 2

def test_id_range_scalar_indexing():
    idr = IdRange(range(10, 20), np.uint32)
    
    val = idr[0]
    assert isinstance(val, np.uint32)
    assert val == 10
    
    val2 = idr[-1]
    assert isinstance(val2, np.uint32)
    assert val2 == 19

def test_id_range_advanced_indexing():
    idr = IdRange(range(100, 110), np.uint32)
    
    # Slice indexing
    sliced = idr[2:5]
    assert isinstance(sliced, np.ndarray)
    assert sliced.dtype == np.uint32
    assert np.array_equal(sliced, np.array([102, 103, 104], dtype=np.uint32))
    
    # List indexing
    listed = idr[[0, -1, 3]]
    assert isinstance(listed, np.ndarray)
    assert np.array_equal(listed, np.array([100, 109, 103], dtype=np.uint32))
    
    # Boolean mask indexing
    mask = [False] * 10
    mask[0] = True
    mask[9] = True
    masked = idr[mask]
    assert isinstance(masked, np.ndarray)
    assert np.array_equal(masked, np.array([100, 109], dtype=np.uint32))

def test_id_range_reify():
    idr = IdRange(range(0, 5), np.uint32)
    arr = idr.reify()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint32
    assert np.array_equal(arr, np.array([0, 1, 2, 3, 4], dtype=np.uint32))
