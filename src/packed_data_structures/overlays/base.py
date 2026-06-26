from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packed_data_structures.table import PackedArrayDB
    from .registry import SchemaRegistry
    from .hooks import TransactionHook


class DbOverlay:
    """Base class for structural abstractions (Overlays).

    An Overlay is responsible for:
    1. Defining the data base Schema (Tables/Columns) via a  Registry.
    2. Creating Runtime Hooks for transactions.
    3. Binding to the live DB instance to provide a High-Level API.
    """

    def register_schema(self, registry: SchemaRegistry) -> None:
        """Phase 1: Define tables and inject columns into the registry.

        Args:
            registry: The central schema registry.
        """
        pass

    def create_hooks(self) -> list[TransactionHook]:
        """Phase 2: Return new hooks for a transaction lifecycle.

        Returns:
            A list of transaction hooks.
        """
        return []

    def bind(self, db: PackedArrayDB) -> None:
        """Phase 3: Bind the overlay to the live database instance.

        Args:
            db: The database instance this overlay abstracts.
        """
        pass


class DbFeature:
    """A modular component attached to an Overlay Layer.

    Can modify the schema (inject columns) and provide runtime hooks.
    """

    # Note: Specific implementations (like GraphFeature) usually subclass this
    # to enforce type safety on the 'layer' argument.
    pass

