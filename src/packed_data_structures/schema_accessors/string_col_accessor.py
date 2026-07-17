from __future__ import annotations

from packed_data_structures.schemas.string_col import StringColSchema
from .schema_accessor import SchemaAccessor
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class StringColSchemaAccessor(SchemaAccessor[StringColSchema, np.str_]):
    """Provides a typed view into a fixed-length string column."""

    pass
