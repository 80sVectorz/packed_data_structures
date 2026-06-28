import pytest
import numpy as np
import time

from packed_data_structures.dirty_tracking import (
    DirtyTrackingArray,
    TimestampRef,
    ProvidesDirtyTimestamp,
    tracked_njit,
)

def test_timestamp_ref():
    ref = TimestampRef(100)
    assert ref.last_dirty_timestamp == 100
    ref.update()
    assert ref.last_dirty_timestamp > 100

def test_dirty_tracking_array_init():
    arr = np.zeros(5, dtype=np.int32)
    dt_arr = DirtyTrackingArray(arr)
    
    assert isinstance(dt_arr, DirtyTrackingArray)
    assert dt_arr.timestamp_ref is not None
    
def test_dirty_tracking_array_mutation():
    dt_arr = DirtyTrackingArray(np.zeros(5, dtype=np.int32))
    initial_ts = dt_arr.last_dirty_timestamp
    
    time.sleep(0.001)
    
    # Mutate via __setitem__
    dt_arr[0] = 10
    
    assert dt_arr.last_dirty_timestamp > initial_ts
    
    ts2 = dt_arr.last_dirty_timestamp
    time.sleep(0.001)
    
    # Mutate via ufunc
    dt_arr += 5
    
    assert dt_arr.last_dirty_timestamp > ts2

class MockDirtySource(ProvidesDirtyTimestamp):
    def __init__(self, arrays):
        super().__init__()
        self.arrays = arrays
        
    def _collect_dirty_sources(self):
        return tuple(self.arrays)

def test_provides_dirty_timestamp_aggregation():
    arr1 = DirtyTrackingArray(np.zeros(5))
    arr2 = DirtyTrackingArray(np.zeros(5))
    
    mock = MockDirtySource([arr1, arr2])
    
    # Initially the max timestamp
    ts1 = mock.last_dirty_timestamp
    
    time.sleep(0.001)
    arr2[0] = 1.0
    
    ts2 = mock.last_dirty_timestamp
    assert ts2 > ts1
    assert ts2 == arr2.last_dirty_timestamp

def test_tracked_njit():
    dt_arr = DirtyTrackingArray(np.zeros(5))
    initial_ts = dt_arr.last_dirty_timestamp
    time.sleep(0.001)
    
    @tracked_njit(mutates=lambda a: (a,))
    def dummy_func(a):
        a[0] = 100
        return a
        
    dummy_func(dt_arr)
    
    assert dt_arr.last_dirty_timestamp > initial_ts
