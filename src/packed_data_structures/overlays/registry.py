from __future__ import annotations
from collections import defaultdict
from packed_data_structures.schemas import TableSchema, ColSchemaLike, IndexSpec


class SchemaRegistry:
    """A smart staging area for defining database structure.

    Allows features to inject columns into tables that are defined elsewhere
    or haven't been created yet.
    """

    def __init__(self):
        self._tables: dict[str, TableSchema] = {}
        # Stores columns waiting for their parent table to be defined
        self._pending_cols: dict[str, list[ColSchemaLike]] = defaultdict(list)

    def register_table(self, schema: TableSchema) -> None:
        """Register an existing TableSchema object.

        Flushes any pending columns waiting for this table name and
        injects them into the schema before it is finalized.

        Args:
            schema: The TableSchema to register.

        Raises:
            ValueError: If a table with the same name is already registered.
        """
        if schema.name in self._tables:
            raise ValueError(f"Table '{schema.name}' is already registered.")

        self._tables[schema.name] = schema

        # Flush pending columns
        if schema.name in self._pending_cols:
            for col in self._pending_cols[schema.name]:
                schema.register_new_column(col)
            del self._pending_cols[schema.name]

    def ensure_table(self, name: str, index_spec: IndexSpec) -> TableSchema:
        """Retrieve a table definition or create it if it doesn't exist.

        Automatically flushes any pending columns into the new table.

        Args:
            name: The name of the table.
            index_spec: The index specification to use if the table must be created.

        Returns:
            The newly created or existing TableSchema.
        """
        if name not in self._tables:
            tbl = TableSchema(name, index_spec, [])
            self._tables[name] = tbl

            # Flush pending columns
            if name in self._pending_cols:
                for col in self._pending_cols[name]:
                    tbl.register_new_column(col)
                del self._pending_cols[name]

        return self._tables[name]

    def add_column(self, table_name: str, col: ColSchemaLike) -> None:
        """Register a column for a specific table.

        If the table exists, the column is added immediately.
        If not, the column is queued until `ensure_table` or `register_table` is called.

        Args:
            table_name: The name of the target table.
            col: The column schema to inject.
        """
        if table_name in self._tables:
            self._tables[table_name].register_new_column(col)
        else:
            self._pending_cols[table_name].append(col)

    def build(self) -> tuple[TableSchema, ...]:
        """Finalizes the schema definition.

        Raises:
            ValueError: if columns are left pending for tables that were never created.
        """
        if self._pending_cols:
            missing_tables = list(self._pending_cols.keys())
            raise ValueError(
                f"Schema definition incomplete. Columns registered for undefined tables: {missing_tables}"
            )
        return tuple(self._tables.values())
