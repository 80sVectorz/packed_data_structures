from __future__ import annotations
from typing import TYPE_CHECKING
from src.packed_data_structures.schemas import ForeignKeySchema
import hashlib

if TYPE_CHECKING:
    from src.packed_data_structures.database import PackedArrayDB
    from src.packed_data_structures.table import PackedArrayTable


class DatabaseVisualizer:
    # Deterministic word lists for easy memorization
    ADJECTIVES = [
        "Swift",
        "Quiet",
        "Crimson",
        "Azure",
        "Golden",
        "Bravo",
        "Neon",
        "Rusty",
        "Fast",
        "Speedy",
        "Big",
        "Small",
        "Square",
        "Round",
        "Scary",
    ]
    NOUNS = [
        "Falcon",
        "Ghost",
        "Cobra",
        "Hammer",
        "Pulse",
        "Vortex",
        "Echo",
        "Titan",
    ]

    @classmethod
    def get_alias(cls, table_name: str, index: int, missing_val: int) -> str:
        """Creates a stable, easy-to-remember name for a specific row."""
        if index == missing_val:
            return "∅"

        seed = f"{table_name}:{index}".encode()
        h = int(hashlib.md5(seed).hexdigest(), 16)
        adj = cls.ADJECTIVES[h % len(cls.ADJECTIVES)]
        noun = cls.NOUNS[(h // len(cls.ADJECTIVES)) % len(cls.NOUNS)]
        return f"{adj}-{noun}"

    @staticmethod
    def format_db(db: PackedArrayDB) -> str:
        lines = [
            f"═══ DB DEBUG VIEW (Transaction Active: {db._transaction_ctx is not None}) ═══",
            "",
        ]
        for table in db.tables:
            lines.append(DatabaseVisualizer._format_table(table))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_table(table: PackedArrayTable) -> str:
        if len(table) == 0:
            return f"Table: {table.name} (Empty / 0 rows)"

        # Identify logical columns, hiding internal _adj_ implementation details
        display_cols = [c for c in table.schema.cols if not c.name.startswith("_adj_")]
        missing = table.schema.index_spec.missing

        col_labels = [""] + [col.name for col in display_cols]
        if table.schema.subscribers:
            col_labels.append("(!) Adjacency / Links")

        rows = []
        for i in range(len(table)):
            entity_label = (
                f"{DatabaseVisualizer.get_alias(table.name, i, missing)} ({i})"
            )
            row_data = [entity_label]

            for col in display_cols:
                val = table.arrays[table.column_ids[col]].view[i]

                if isinstance(col, ForeignKeySchema):
                    target_table = table.db.get_table(col.target_table.name)
                    missing = target_table.schema.index_spec.missing

                    target_name = entity_label = (
                        f"{DatabaseVisualizer.get_alias(table.name, val, missing)} ({i})"
                    )

                    if val == missing:
                        row_data.append("∅ (null)")
                    elif 0 <= val < len(target_table):
                        row_data.append(f"→ {target_table.name}[{target_name}]")
                    else:
                        # EXPLICIT ERROR: Index out of bounds
                        row_data.append(f"!! OOB[{val}] !!")
                else:
                    row_data.append(str(val))

            # Adjacency Diagnostics
            if table.schema.subscribers:
                row_data.append(DatabaseVisualizer._check_adj_health(table, i))

            rows.append(row_data)

        return DatabaseVisualizer._tabulate(table.name, col_labels, rows)

    @staticmethod
    def _check_adj_health(table: PackedArrayTable, head_idx: int) -> str:
        """Manually traces adjacency lists to detect corruption (cycles/OOB)."""
        reports = []
        for fk in table.schema.subscribers:
            parent_table = table.db.get_table(fk.parent_table.name)
            vw_head = table[fk.adj_head].view
            vw_next = parent_table[fk.adj_next].view
            missing = table.schema.index_spec.missing

            chain = []
            visited = set()
            curr = vw_head[head_idx]

            error = None
            while curr != missing:
                if curr < 0 or curr >= len(parent_table):
                    error = f"OOB_LINK({curr})"
                    break
                if curr in visited:
                    error = "CYCLE_DETECTED"
                    break

                visited.add(curr)
                chain.append(curr)
                curr = vw_next[curr]

            # Compare against track_counts if enabled
            if fk.adjacency_conf.track_counts:
                actual_count = len(chain)
                stored_count = table[fk.adj_count].view[head_idx]
                if actual_count != stored_count:
                    error = (
                        f"COUNT_MISMATCH(got {actual_count}, expected {stored_count})"
                    )

            status = f"{fk.parent_table.name}{chain}"
            if error:
                status = f"!! {status} ERR: {error} !!"
            reports.append(status)

        return " | ".join(reports)

    @staticmethod
    def _tabulate(table_name: str, col_labels: list[str], rows: list[list[str]]) -> str:
        widths = [len(label) for label in col_labels]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        title_sep = f"{table_name:─^{sum(widths)+2*(len(widths)-1)}}"
        rows_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        sep = "─" * len(title_sep)
        result = [title_sep, rows_fmt.format(*col_labels), sep]
        for row in rows:
            result.append(rows_fmt.format(*row))
        return "\n".join(result)
