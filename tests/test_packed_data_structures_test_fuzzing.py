import pytest
import numpy as np
from hypothesis import reproduce_failure, settings, strategies as st
from hypothesis import note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    initialize,
    invariant,
)
from dataclasses import dataclass

from packed_data_structures.db_visualizer import DatabaseVisualizer
from packed_data_structures.database import PackedArrayDB
from packed_data_structures.schemas import (
    TableSchema,
    IndexSpec,
    DataColSchema,
    ForeignKeySchema,
    AdjacencyListConf,
)
from packed_data_structures.transaction_context import TransactionContextException


@dataclass(frozen=True)
class TargetModel:
    uid: int
    payload: int


@dataclass(frozen=True)
class SourceModel:
    uid: int
    payload: int
    target_uid: int


@reproduce_failure(
    "6.150.0",
    b"AXiclVVpTxRBEO3aHXYX3ZX7UBBBBASJdwQU0apg8IjgRYzIocYf6xc1XjH6RU00MYboFz9oNMbX9WZgJ6iJsMubrumuelX1qgnBwrsgOiPaIrooOil6SXRMdJfooGg33oX4K2qEJcIkYYRwk5DuPEw4WGfEj+iyaB9gjvY2wlzuaC23OkVYzBnPuz/R//tsPiFhC49NeqMt5nc/574JIBvZpuyOb9kStvjNwMJ7kyFP+KToEGCAx8YJZwlThNuEZcLdXJwqoS8L96e06h4tfPD03uD7JH13SLRn028T4R7hzGbv5J+lc08HRW+Ai5vWwFt03p9HRI9uUt5NOE1YyeXYSuglTKRZ/T30lrXJgIV1E7Hw0GQ4badUPetPznIF3yEGGMuVO+2B5rScl9lSvV5Fm0UbIj+AaFm0QErbsUDQHgsvRRMfn2i+5JU4gnjZMOF80RtwFq/3w2lNtFPjGXRWdJvoDnda3siw5NaGLGDZLdF50WlIRqPglmJ9YaqQnIVXJkULb03KFj6b9Ft4ZOGbhZ+Zi0JdzVDFp6D/wKv3ArxMGj2VVSjEO40GX4ZoRPfCOq/u4QQSweFuQIIX6hzj1JZiu536MRXqyuMN8dx1ZTlRhCqg0T+JvyqlZs9eWN5eULLwC1129uW6pKXugfP8Hd+vMEURmOzDXpMCnuIpC6+jOBDBwjM8vUXZY5ILol0YfZZ/gRxnOfqzJB5nBrmPcjWMFRiv0bjMA+N81wljzK2CKMEtNU+v4jQb3ZLUdbGYiomtGMWW+HjRr90ZV5j5MSwnMEtoQZwz9TI0s7jXGLmV9e9XD7YAHtDULfUQ01gVNzi2pMpr8Z1lsk3cV8F9JFlBC5muGlGuH4EjAB3cEb2Q3hqt0DWgk7pOeEXt5MXSHgkgcvB2YtWWJV/yurS6oqqejAfJwhUz6SfpjbOU3otdcYi5Gatu3idtMSVvySqz7ueWK8qRoaA6sCtdFWPjTTpNKPeSZ9qQDdRG+pTUF3zXKRz8/RgvdTg0wbZmSApUUJsoKqnB6YjznUaxQSdwLOBzMNYoxLjuPGHBOehNFAMcP8cMQr8mXel7kz0QJybYvR4JHLr2eKs6vVVCMycx/vfqiCPoxh7CFJu0NyVQ8vsMjivu8qroOewRPRB4F1VcF1Un1uHUt3Me432HI78BT64QpA==",
)
class DatabaseIntegrityMachine(RuleBasedStateMachine):
    run_counter = 0

    def __init__(self):
        super().__init__()
        if DatabaseIntegrityMachine.run_counter == 0:
            # Move to a new line on the very first run so we don't overwrite pytest's "collected X items"
            print(flush=True)

        DatabaseIntegrityMachine.run_counter += 1
        # \r prints over the same line (inplace)
        print(f"\rTest Example #{self.run_counter}", end="", flush=True)

        self.model_targets: dict[int, TargetModel] = {}
        self.model_sources: dict[int, SourceModel] = {}
        self.next_id = 1

    @initialize()
    def init_db(self):
        idx_spec = IndexSpec.from_dtype(np.int32)

        # --- Schema Definition ---
        # Targets: The referenced entities
        self.t_uid_col = DataColSchema("_uid", np.int32, default=-1)
        self.t_payload_col = DataColSchema("payload", np.int32)
        self.t_schema = TableSchema(
            "targets", idx_spec, [self.t_uid_col, self.t_payload_col]
        )

        # Sources: The referencing entities
        self.s_uid_col = DataColSchema("_uid", np.int32, default=-1)
        self.s_payload_col = DataColSchema("payload", np.int32)

        adj_conf = AdjacencyListConf(track_counts=True)
        # Foreign Key on Source pointing to Target
        self.fk_schema = ForeignKeySchema(
            "target_ref", self.t_schema, adjacency_conf=adj_conf
        )

        self.s_schema = TableSchema(
            "sources", idx_spec, [self.s_uid_col, self.s_payload_col, self.fk_schema]
        )

        self.db = PackedArrayDB(self.t_schema, self.s_schema)
        self.t_targets = self.db.get_table(self.t_schema)
        self.t_sources = self.db.get_table(self.s_schema)

    def _get_phys_indices_map(self, table, uid_col):
        """Helper to map UIDs to current physical indices in the DB."""
        uids = table[uid_col].view
        return {uid: idx for idx, uid in enumerate(uids) if uid != -1}

    @rule(
        target_payloads=st.lists(st.integers(0, 100), min_size=1, max_size=10),
        source_data=st.data(),
    )
    def apply_batch_transaction(self, target_payloads, source_data):
        """A complex rule that performs a mixed batch of:
        1. Adding new targets.
        2. Adding new sources (some pointing to existing targets, some to NEW targets).
        3. Updating existing targets.
        4. Updating existing sources (payloads and re-linking FKs).
        5. Deleting existing targets (triggering cascade).
        6. Deleting existing sources.
        """  # noqa: D205
        # --- 1. Draw Strategies ---
        new_target_defs = [{"payload": p} for p in target_payloads]
        existing_t_uids = list(self.model_targets.keys())
        num_new_targets = len(new_target_defs)

        # Draw new sources
        new_source_defs = source_data.draw(
            st.lists(
                st.tuples(
                    st.integers(0, 100),  # Payload
                    st.one_of(
                        st.sampled_from(existing_t_uids)
                        if existing_t_uids
                        else st.nothing(),  # Existing UID
                        st.integers(
                            0, num_new_targets - 1
                        ),  # Index into new_target_defs
                    ),
                ),
                min_size=0,
                max_size=10,
            )
        )

        # Draw Deletes
        to_del_t_uids = set(
            source_data.draw(st.lists(st.sampled_from(existing_t_uids), unique=True))
            if existing_t_uids
            else []
        )
        existing_s_uids = list(self.model_sources.keys())
        to_del_s_uids = set(
            source_data.draw(st.lists(st.sampled_from(existing_s_uids), unique=True))
            if existing_s_uids
            else []
        )

        # Draw Updates (Disjoint from Deletes for sanity, though DB should catch overlaps)
        available_for_upd_t = [u for u in existing_t_uids if u not in to_del_t_uids]
        to_upd_t_uids = (
            source_data.draw(
                st.lists(st.sampled_from(available_for_upd_t), unique=True)
            )
            if available_for_upd_t
            else []
        )
        updates_t = {
            uid: source_data.draw(st.integers(200, 300)) for uid in to_upd_t_uids
        }

        available_for_upd_s = [u for u in existing_s_uids if u not in to_del_s_uids]
        to_upd_s_uids = (
            source_data.draw(
                st.lists(st.sampled_from(available_for_upd_s), unique=True)
            )
            if available_for_upd_s
            else []
        )

        valid_targets_for_retarget = [
            t for t in existing_t_uids if t not in to_del_t_uids
        ]
        updates_s = {}
        for uid in to_upd_s_uids:
            new_pay = source_data.draw(st.integers(200, 300))
            new_target = None
            if valid_targets_for_retarget and source_data.draw(st.booleans()):
                new_target = source_data.draw(
                    st.sampled_from(valid_targets_for_retarget)
                )
            updates_s[uid] = (new_pay, new_target)

        # --- 2. Prepare Data (Assign UIDs tentatively) ---
        # We don't apply these to self.model_... yet.
        temp_next_id = self.next_id

        # Map: NewTargetIndex -> AssignedUID
        new_t_uid_map = {}
        for i in range(len(new_target_defs)):
            new_t_uid_map[i] = temp_next_id
            temp_next_id += 1

        # Map: NewSourceIndex -> AssignedUID
        # Store tuple needed for Model update later: (UID, Payload, ResolvedTargetUID)
        new_s_model_data = []
        start_s_uid = temp_next_id

        for i, (payload, target_ref) in enumerate(new_source_defs):
            current_uid = start_s_uid + i

            # Resolve Target UID for the Model
            if isinstance(target_ref, int) and target_ref < num_new_targets:
                actual_target_uid = new_t_uid_map[target_ref]
            else:
                actual_target_uid = target_ref

            new_s_model_data.append((current_uid, payload, actual_target_uid))

        temp_next_id += len(new_source_defs)

        # --- 3. Execute Transaction on DB ---
        # We fetch physical indices based on CURRENT state
        phys_map_t = self._get_phys_indices_map(self.t_targets, self.t_uid_col)
        phys_map_s = self._get_phys_indices_map(self.t_sources, self.s_uid_col)

        try:
            with self.db.transaction():
                # A. Register Additions (Targets)
                t_records = []
                for i in range(num_new_targets):
                    t_records.append(
                        {
                            self.t_uid_col: new_t_uid_map[i],
                            self.t_payload_col: new_target_defs[i]["payload"],
                        }
                    )
                # We need the virtual range to link new sources to new targets
                virtual_t_ids = self.t_targets.add_entries(t_records, "row_major")
                virtual_start = virtual_t_ids.start if virtual_t_ids else 0

                # B. Register Additions (Sources)
                s_records = []
                for i, (payload, target_ref) in enumerate(new_source_defs):
                    current_uid = (
                        start_s_uid + i
                    )  # Re-calc UID to match preparation step

                    # Resolve FK: Physical Index (existing) or Virtual Index (new)
                    if isinstance(target_ref, int) and target_ref < num_new_targets:
                        # Pointing to a new target -> Use Virtual Index
                        fk_val = virtual_start + target_ref
                    else:
                        # Pointing to existing target -> Use Physical Index
                        if target_ref in phys_map_t:
                            fk_val = phys_map_t[target_ref]
                        else:
                            # If strategy picked a target that doesn't exist (shouldn't happen), skip
                            continue

                    s_records.append(
                        {
                            self.s_uid_col: current_uid,
                            self.s_payload_col: payload,
                            self.fk_schema: fk_val,
                        }
                    )

                if s_records:
                    self.t_sources.add_entries(s_records, "row_major")

                # C. Register Deletes
                del_t_idxs = [phys_map_t[u] for u in to_del_t_uids if u in phys_map_t]
                del_s_idxs = [phys_map_s[u] for u in to_del_s_uids if u in phys_map_s]

                if del_t_idxs:
                    self.t_targets.del_entries(del_t_idxs)
                if del_s_idxs:
                    self.t_sources.del_entries(del_s_idxs)

                # D. Register Updates
                # Targets
                t_upd_indices = []
                t_upd_values = []
                for uid, pay in updates_t.items():
                    if uid in phys_map_t:
                        t_upd_indices.append(phys_map_t[uid])
                        t_upd_values.append(pay)

                if t_upd_indices:
                    self.t_targets.update_entries(
                        {self.t_payload_col: (t_upd_indices, t_upd_values)}
                    )

                # Sources
                s_pay_indices = []
                s_pay_values = []
                s_fk_indices = []
                s_fk_values = []

                for uid, (pay, new_tgt_ref) in updates_s.items():
                    if uid in phys_map_s:
                        phys_idx = phys_map_s[uid]
                        # Update Payload
                        s_pay_indices.append(phys_idx)
                        s_pay_values.append(pay)

                        # Update FK if requested
                        if new_tgt_ref is not None:
                            if new_tgt_ref in phys_map_t:
                                s_fk_indices.append(phys_idx)
                                s_fk_values.append(phys_map_t[new_tgt_ref])

                if s_pay_indices:
                    self.t_sources.update_entries(
                        {self.s_payload_col: (s_pay_indices, s_pay_values)}
                    )
                if s_fk_indices:
                    self.t_sources.update_entries(
                        {self.fk_schema: (s_fk_indices, s_fk_values)}
                    )

        except TransactionContextException as e:
            # The DB rejected the transaction (e.g., updating a deleted row).
            # We catch this, Note it, and RETURN.
            # Crucially, we do NOT update self.model_... so the Test State matches the DB State.
            note(f"Transaction Rejected by DB (Correctly): {e}")
            return

        # --- 4. Update Model (Only reached if DB success) ---

        # Commit the ID counter
        self.next_id = temp_next_id

        # Apply New Targets
        for i, val in enumerate(new_target_defs):
            uid = new_t_uid_map[i]
            self.model_targets[uid] = TargetModel(uid, val["payload"])

        # Apply New Sources
        for uid, pay, tgt_uid in new_s_model_data:
            self.model_sources[uid] = SourceModel(uid, pay, tgt_uid)

        # Apply Deletes (Cascade)
        # 1. Explicit Source deletes
        for uid in to_del_s_uids:
            if uid in self.model_sources:
                del self.model_sources[uid]

        # 2. Target deletes (Cascade to Sources)
        for uid in to_del_t_uids:
            if uid in self.model_targets:
                del self.model_targets[uid]
                # Find sources pointing here
                cascade_s = [
                    s.uid for s in self.model_sources.values() if s.target_uid == uid
                ]
                for s_uid in cascade_s:
                    if s_uid in self.model_sources:
                        del self.model_sources[s_uid]

        # Apply Updates
        for uid, pay in updates_t.items():
            if uid in self.model_targets:
                self.model_targets[uid] = TargetModel(uid, pay)

        for uid, (pay, new_tgt_ref) in updates_s.items():
            if uid in self.model_sources:
                curr = self.model_sources[uid]
                tgt = new_tgt_ref if new_tgt_ref is not None else curr.target_uid
                self.model_sources[uid] = SourceModel(uid, pay, tgt)

        # --- Logging ---
        note("--- Transaction Batch Applied ---")
        if new_target_defs:
            note(f"Added {len(new_target_defs)} Targets:\n{new_target_defs}")
        if new_source_defs:
            note(f"Added {len(new_source_defs)} Sources:\n{new_source_defs}")
        if to_del_t_uids:
            note(f"Deleted {len(to_del_t_uids)} Targets:\n{to_del_t_uids}")
        if to_del_s_uids:
            note(f"Deleted {len(to_del_s_uids)} Sources:\n{to_del_s_uids}")
        note(DatabaseVisualizer.format_db(self.db))
        note("-" * 30)

    @invariant()
    def check_consistency(self):
        # 1. Check Counts
        assert (
            len(self.t_targets) == len(self.model_targets)
        ), f"Target count mismatch. DB: {len(self.t_targets)}, Model: {len(self.model_targets)}"
        assert (
            len(self.t_sources) == len(self.model_sources)
        ), f"Source count mismatch. DB: {len(self.t_sources)}, Model: {len(self.model_sources)}"

        # 2. Verify Targets Data
        db_t_uids = self.t_targets[self.t_uid_col].view
        db_t_payloads = self.t_targets[self.t_payload_col].view

        for i in range(len(self.t_targets)):
            uid = db_t_uids[i]
            payload = db_t_payloads[i]

            assert uid in self.model_targets, f"Found unexpected Target UID {uid} in DB"
            assert (
                self.model_targets[uid].payload == payload
            ), f"Payload mismatch for Target {uid}"

        # 3. Verify Sources Data & Foreign Keys
        db_s_uids = self.t_sources[self.s_uid_col].view
        db_s_payloads = self.t_sources[self.s_payload_col].view
        db_s_fks = self.t_sources[self.fk_schema].view

        # Access Target UIDs to resolve FK pointers
        target_uid_lookup = self.t_targets[self.t_uid_col].view

        for i in range(len(self.t_sources)):
            s_uid = db_s_uids[i]
            s_payload = db_s_payloads[i]
            s_fk_idx = db_s_fks[i]

            assert (
                s_uid in self.model_sources
            ), f"Found unexpected Source UID {s_uid} in DB"
            model_rec = self.model_sources[s_uid]

            assert (
                model_rec.payload == s_payload
            ), f"Payload mismatch for Source {s_uid}"

            # Check for dangling pointer
            if s_fk_idx == self.t_schema.index_spec.missing:
                pytest.fail(
                    f"Found dangling FK (Set to Missing) at Source {s_uid}. Expected Cascade Delete."
                )

            # Resolve FK pointer
            if s_fk_idx < 0 or s_fk_idx >= len(target_uid_lookup):
                pytest.fail(f"FK index out of bounds for Source {s_uid}: {s_fk_idx}")

            actual_target_uid = target_uid_lookup[s_fk_idx]
            expected_target_uid = model_rec.target_uid

            assert (
                actual_target_uid == expected_target_uid
            ), f"FK Mismatch for Source {s_uid}. Points to Target {actual_target_uid}, expected {expected_target_uid}"


DatabaseIntegrityMachine.TestCase.settings = settings(
    print_blob=True, max_examples=1000
)
TestDBIntegrity = DatabaseIntegrityMachine.TestCase
