from collections import UserDict
from dataclasses import dataclass
from enum import Enum


AtomicType = Enum(
    "AtomicType",
    [
        "uint8",
        "uint16",
        "int16",
        "int32",
        "float32",
        "float64",
        "bytes",
        "undefined",
    ],
)


@dataclass
class TokenProperties:
    """Class to hold properties of a token.
    Attributes:
        location (int): The location of the token in the binary file.
        atomic_type (AtomicType): The type of the token.
    """
    location: int = -1
    atomic_type: AtomicType = AtomicType.undefined


BinaryMetaDataStructure = UserDict(
    {
        "template_name": TokenProperties(location=16, atomic_type=AtomicType.bytes),
        "comm_type": TokenProperties(location=32, atomic_type=AtomicType.uint16),
        "wave_descriptor": TokenProperties(location=36, atomic_type=AtomicType.int32),
        "user_text": TokenProperties(location=40, atomic_type=AtomicType.int32),
        "trig_time_array": TokenProperties(location=48, atomic_type=AtomicType.int32),
        "wave_array1": TokenProperties(location=60, atomic_type=AtomicType.int32),
        "instrument_name": TokenProperties(location=76, atomic_type=AtomicType.bytes),
        "instrument_number": TokenProperties(location=92, atomic_type=AtomicType.int32),
        "trace_label": TokenProperties(),
        "wave_array_count": TokenProperties(location=116, atomic_type=AtomicType.int32),
        "vertical_gain": TokenProperties(location=156, atomic_type=AtomicType.float32),
        "vertical_offset": TokenProperties(location=160, atomic_type=AtomicType.float32),
        "nominal_bits": TokenProperties(location=172, atomic_type=AtomicType.uint16),
        "horiz_interval": TokenProperties(location=176, atomic_type=AtomicType.float32),
        "horiz_offset": TokenProperties(location=180, atomic_type=AtomicType.float64),
        "vert_unit": TokenProperties(),
        "hor_unit": TokenProperties(),
        "sequence_segments": TokenProperties(location=144, atomic_type=AtomicType.int32),
        "trigger_seconds": TokenProperties(location=296, atomic_type=AtomicType.float64),
        "trigger_minutes": TokenProperties(location=304, atomic_type=AtomicType.uint8),
        "trigger_hours": TokenProperties(location=305, atomic_type=AtomicType.uint8),
        "trigger_days": TokenProperties(location=306, atomic_type=AtomicType.uint8),
        "trigger_months": TokenProperties(location=307, atomic_type=AtomicType.uint8),
        "trigger_years": TokenProperties(location=308, atomic_type=AtomicType.int16),
        "record_type": TokenProperties(location=316, atomic_type=AtomicType.uint16),
        "processing_done": TokenProperties(location=318, atomic_type=AtomicType.uint16),
        "time_base": TokenProperties(location=324, atomic_type=AtomicType.uint16),
        "vertical_coupling": TokenProperties(location=326, atomic_type=AtomicType.uint16),
        "bandwidth_limit": TokenProperties(location=334, atomic_type=AtomicType.uint16),
        "wave_source": TokenProperties(location=344, atomic_type=AtomicType.uint16),
    }
)