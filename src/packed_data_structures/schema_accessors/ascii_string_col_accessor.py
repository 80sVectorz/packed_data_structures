from __future__ import annotations

from packed_data_structures.schemas.ascii_string_col import AsciiStringColSchema
from .schema_accessor import SchemaAccessor
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class AsciiStringColSchemaAccessor(SchemaAccessor[AsciiStringColSchema, np.bytes_]):
    """Provides a typed view into a fixed-length ASCII byte-string column."""

    pass
