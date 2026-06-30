import pytest
import numpy as np


def test_single_tracker_basic(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    with populated_db.transaction() as ctx:
        # We delete row 0, so row 2 will swap into row 0.
        # We track row 2 to see where it ends up.
        tracker = ctx.create_tracker(table_a_schema, 2)

        # Pre-commit checks
        with pytest.raises(RuntimeError):
            tracker.reify()

        ctx.register_deletions(table_a_schema, [0])

    # Post-commit checks
    # row 2 moved to 0
    assert tracker.reify() == 0
    assert (
        type(tracker.reify()) == table_a_schema.index_spec.dtype
    )  # should be same as index spec dtype
    assert tracker.query(2) == 0

    # query_array
    np.testing.assert_array_equal(tracker.query_array(np.array([2])), np.array([0]))

    # Test unpacking/iter
    (unpacked,) = tracker
    assert unpacked == 0


def test_array_tracker_swaps(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    # 3 entries exist: 0, 1, 2
    with populated_db.transaction() as ctx:
        # delete 0 -> 2 swaps to 0. 1 stays at 1.
        tracker = ctx.create_tracker(table_a_schema, [1, 2])
        ctx.register_deletions(table_a_schema, [0])

    reified = tracker.reify()
    np.testing.assert_array_equal(reified, np.array([1, 0]))
    assert reified.dtype == np.uint32

    assert tracker.query(2) == 0
    assert tracker.query(1) == 1

    # Unpacking
    a, b = tracker
    assert a == 1
    assert b == 0

    # Slicing
    assert tracker[1] == 0
    np.testing.assert_array_equal(tracker[0:2], np.array([1, 0]))


@pytest.mark.parametrize("storage", ["hard", "soft", "auto"])
def test_range_tracker_additions(populated_db, base_schemas, storage):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)
    col_weight = next(c for c in table_a_schema.cols if c.name == "weight")

    with populated_db.transaction() as ctx:
        # 3 exist, we add 5. staged_ids = range(3, 8)
        staged = table_a.add_entries(
            records={col_weight: [10, 20, 30, 40, 50]}, records_shape="col_major"
        )
        # We also delete 0, so one hole is created.
        ctx.register_deletions(table_a_schema, [0])
        # Additions fill: 0 (hole), 3, 4, 5, 6

        tracker = ctx.create_tracker(table_a_schema, staged, storage_method=storage)

    expected_physical = np.array([0, 3, 4, 5, 6], dtype=np.uint32)
    np.testing.assert_array_equal(tracker.reify(), expected_physical)

    assert tracker.query(3) == 0
    assert tracker.query(7) == 6

    a, b, c, d, e = tracker
    assert a == 0
    assert e == 6


def test_missing_indices(populated_db, base_schemas):
    table_a_schema, _ = base_schemas
    table_a = populated_db.get_table(table_a_schema)

    with populated_db.transaction() as ctx:
        tracker = ctx.create_tracker(table_a_schema, [0, 1])
        ctx.register_deletions(table_a_schema, [1])  # delete 1

    # 0 stays 0. 1 is missing
    missing = table_a_schema.index_spec.missing
    np.testing.assert_array_equal(tracker.reify(), np.array([0, missing]))
    assert tracker.query(1) == missing
