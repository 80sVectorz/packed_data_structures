import numpy as np


def test_pm_fk_additions(pm_populated_db, pm_base_schemas):
    table_a_schema, table_b_schema, table_c_schema = pm_base_schemas
    table_a = pm_populated_db.get_table(table_a_schema)
    table_b = pm_populated_db.get_table(table_b_schema)
    table_c = pm_populated_db.get_table(table_c_schema)

    pm_fk = next(c for c in table_c_schema.cols if c.name == "link_to_a_or_b")
    type_a = pm_fk.type_id_mapping[table_a_schema]
    type_b = pm_fk.type_id_mapping[table_b_schema]

    # Check additions were correct
    assert len(table_c) == 5
    assert np.array_equal(table_c[pm_fk][:], [0, 0, 1, 0, 1])
    assert np.array_equal(
        table_c[pm_fk.type_id_col][:], [type_a, type_a, type_a, type_b, type_b]
    )

    adj_head_a = pm_fk.adj_head_columns[type_a]
    adj_count_a = pm_fk.adj_count_columns[type_a]

    # A[0] has C[0], C[1]
    assert table_a[adj_head_a][0] in (0, 1)  # Head can be 0 or 1
    assert table_a[adj_count_a][0] == 2
    # A[1] has C[2]
    assert table_a[adj_head_a][1] == 2
    assert table_a[adj_count_a][1] == 1

    adj_head_b = pm_fk.adj_head_columns[type_b]
    adj_count_b = pm_fk.adj_count_columns[type_b]

    # B[0] has C[3]
    assert table_b[adj_head_b][0] == 3
    assert table_b[adj_count_b][0] == 1
    # B[1] has C[4]
    assert table_b[adj_head_b][1] == 4
    assert table_b[adj_count_b][1] == 1

    # Check next/prev links for Table C
    missing = pm_fk.keys_idx_spec.missing
    # C[0] and C[1] should point to each other
    assert table_c[pm_fk.adj_next][0] == 1 or table_c[pm_fk.adj_prev][0] == 1
    assert table_c[pm_fk.adj_next][1] == 0 or table_c[pm_fk.adj_prev][1] == 0
    # C[2], C[3], C[4] are isolated
    assert table_c[pm_fk.adj_next][2] == missing
    assert table_c[pm_fk.adj_prev][2] == missing


def test_pm_fk_update_target_same_table(pm_populated_db, pm_base_schemas):
    table_a_schema, table_b_schema, table_c_schema = pm_base_schemas
    table_a = pm_populated_db.get_table(table_a_schema)
    table_c = pm_populated_db.get_table(table_c_schema)
    pm_fk = next(c for c in table_c_schema.cols if c.name == "link_to_a_or_b")

    type_a = pm_fk.type_id_mapping[table_a_schema]
    adj_head_a = pm_fk.adj_head_columns[type_a]
    adj_count_a = pm_fk.adj_count_columns[type_a]

    with pm_populated_db.transaction():
        # Update C[0] to point to A[1] instead of A[0]
        # It's currently Type A, keeping it Type A.
        # But `allows_updates` on pm_fk type_id_col handles permission,
        # we still must use register_updates_col_major if we are updating internally,
        # but here we can just update the pm_fk column directly.
        table_c.update_entries({pm_fk: {0: 1}})

    assert table_c[pm_fk][0] == 1
    # A[0] should now only have C[1]
    assert table_a[adj_count_a][0] == 1
    assert table_a[adj_head_a][0] == 1
    # A[1] should now have C[2] and C[0]
    assert table_a[adj_count_a][1] == 2
    assert table_a[adj_head_a][1] in (0, 2)


def test_pm_fk_update_type(pm_populated_db, pm_base_schemas):
    table_a_schema, table_b_schema, table_c_schema = pm_base_schemas
    table_a = pm_populated_db.get_table(table_a_schema)
    table_b = pm_populated_db.get_table(table_b_schema)
    table_c = pm_populated_db.get_table(table_c_schema)

    pm_fk = next(c for c in table_c_schema.cols if c.name == "link_to_a_or_b")
    type_b = pm_fk.type_id_mapping[table_b_schema]

    type_a = pm_fk.type_id_mapping[table_a_schema]
    adj_head_a = pm_fk.adj_head_columns[type_a]
    adj_count_a = pm_fk.adj_count_columns[type_a]

    adj_head_b = pm_fk.adj_head_columns[type_b]
    adj_count_b = pm_fk.adj_count_columns[type_b]

    with pm_populated_db.transaction():
        # Update C[0] (which is A[0]) to point to B[1] (type B)
        table_c.update_entries({pm_fk: {0: 1}, pm_fk.type_id_col: {0: type_b}})

    assert table_c[pm_fk][0] == 1
    assert table_c[pm_fk.type_id_col][0] == type_b

    # A[0] lost C[0], should only have C[1] now
    assert table_a[adj_count_a][0] == 1
    assert table_a[adj_head_a][0] == 1

    # B[1] gained C[0], should have C[4] and C[0] now
    assert table_b[adj_count_b][1] == 2
    assert table_b[adj_head_b][1] in (0, 4)

    # C[0] and C[4] should be linked
    assert table_c[pm_fk.adj_next][0] == 4 or table_c[pm_fk.adj_prev][0] == 4


def test_pm_fk_target_deletion_cascade(pm_populated_db, pm_base_schemas):
    table_a_schema, table_b_schema, table_c_schema = pm_base_schemas
    table_a = pm_populated_db.get_table(table_a_schema)
    table_c = pm_populated_db.get_table(table_c_schema)
    pm_fk = next(c for c in table_c_schema.cols if c.name == "link_to_a_or_b")

    # A[0] is linked by C[0] and C[1]
    # Delete A[0]. Because on_delete is CASCADE, C[0] and C[1] should be deleted.
    with pm_populated_db.transaction():
        table_a.del_entry(0)

    # A[0] is deleted. A[1] moves to A[0].
    assert len(table_a) == 1

    # C[0] and C[1] are deleted. Swap-and-pop means C[4] -> C[0], C[3] -> C[1]. C[2] stays at 2.
    assert len(table_c) == 3

    type_a = pm_fk.type_id_mapping[table_a_schema]
    type_b = pm_fk.type_id_mapping[table_b_schema]

    # C[0] is old C[4], pointing to B[1] (which hasn't moved)
    assert table_c[pm_fk.type_id_col][0] == type_b
    assert table_c[pm_fk][0] == 1

    # C[1] is old C[3], pointing to B[0] (which hasn't moved)
    assert table_c[pm_fk.type_id_col][1] == type_b
    assert table_c[pm_fk][1] == 0

    # C[2] is old C[2], pointing to old A[1]. Old A[1] moved to A[0]!
    assert table_c[pm_fk.type_id_col][2] == type_a
    assert table_c[pm_fk][2] == 0


def test_pm_fk_source_deletion(pm_populated_db, pm_base_schemas):
    table_a_schema, table_b_schema, table_c_schema = pm_base_schemas
    table_a = pm_populated_db.get_table(table_a_schema)
    table_c = pm_populated_db.get_table(table_c_schema)
    pm_fk = next(c for c in table_c_schema.cols if c.name == "link_to_a_or_b")

    type_a = pm_fk.type_id_mapping[table_a_schema]
    adj_head_a = pm_fk.adj_head_columns[type_a]
    adj_count_a = pm_fk.adj_count_columns[type_a]

    # Delete C[0], which points to A[0].
    # C[0] and C[1] originally point to A[0].
    with pm_populated_db.transaction():
        table_c.del_entry(0)

    # C[4] moved to C[0].
    assert len(table_c) == 4

    # A[0] should now only point to C[1].
    # But wait, did C[1] move? No.
    assert table_a[adj_count_a][0] == 1
    assert table_a[adj_head_a][0] == 1
