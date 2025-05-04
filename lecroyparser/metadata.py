"""
LeCroy Parser
This module provides the MetaData class, which is used to store metadata information
from LeCroy oscilloscopes. The class is a named tuple that contains various attributes
related to the oscilloscope data, such as template name, communication type, wave descriptor,
user text, trigger time array, wave array, instrument name, instrument number, trace label,
wave array count, vertical gain, vertical offset, nominal bits, horizontal interval,
horizontal offset, vertical unit, horizontal unit, sequence segments, trigger time,
record type, processing done, time base, vertical coupling, bandwidth limit, and wave source.
"""

from enum import Enum
from collections import namedtuple, UserDict

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
    ],
)

MetaTokenProperties = namedtuple(
    "MetaTokenProperties",
    [
      "location",
      "atomic_type",
    ],
)

BinaryMetaDataStructure = UserDict(
    {
        "template_name": MetaTokenProperties(location=16, atomic_type=AtomicType.bytes),
        "comm_type": MetaTokenProperties(location=32, atomic_type=AtomicType.uint16),
        "wave_descriptor": MetaTokenProperties(location=36, atomic_type=AtomicType.int32),
        "user_text": MetaTokenProperties(location=40, atomic_type=AtomicType.int32),
        "trig_time_array": MetaTokenProperties(location=48, atomic_type=AtomicType.int32),
        "wave_array1": MetaTokenProperties(location=60, atomic_type=AtomicType.int32),
        "instrument_name": MetaTokenProperties(location=76, atomic_type=AtomicType.bytes),
        "instrument_number": MetaTokenProperties(location=92, atomic_type=AtomicType.int32),
        "trace_label": "NOT PARSED",
        "wave_array_count": MetaTokenProperties(location=116, atomic_type=AtomicType.int32),
        "vertical_gain": MetaTokenProperties(location=156, atomic_type=AtomicType.float32),
        "vertical_offset": MetaTokenProperties(location=160, atomic_type=AtomicType.float32),
        "nominal_bits": MetaTokenProperties(location=172, atomic_type=AtomicType.uint16),
        "horiz_interval": MetaTokenProperties(location=176, atomic_type=AtomicType.float32),
        "horiz_offset": MetaTokenProperties(location=180, atomic_type=AtomicType.float64),
        "vert_unit": "NOT PARSED",
        "hor_unit": "NOT PARSED",
        "sequence_segments": MetaTokenProperties(location=144, atomic_type=AtomicType.int32),
        "trigger_seconds": MetaTokenProperties(location=296, atomic_type=AtomicType.float64),
        "trigger_minutes": MetaTokenProperties(location=304, atomic_type=AtomicType.uint8),
        "trigger_hours": MetaTokenProperties(location=305, atomic_type=AtomicType.uint8),
        "trigger_days": MetaTokenProperties(location=306, atomic_type=AtomicType.uint8),
        "trigger_months": MetaTokenProperties(location=307, atomic_type=AtomicType.uint8),
        "trigger_years": MetaTokenProperties(location=308, atomic_type=AtomicType.int16),
        "record_type": MetaTokenProperties(location=316, atomic_type=AtomicType.uint16),
        "processing_done": MetaTokenProperties(location=318, atomic_type=AtomicType.uint16),
        "time_base": MetaTokenProperties(location=324, atomic_type=AtomicType.uint16),
        "vertical_coupling": MetaTokenProperties(location=326, atomic_type=AtomicType.uint16),
        "bandwidth_limit": MetaTokenProperties(location=334, atomic_type=AtomicType.uint16),
        "wave_source": MetaTokenProperties(location=344, atomic_type=AtomicType.uint16),
    }
)

MetaData = namedtuple(
    "MetaData",
    [
        "templateName",
        "commType",
        "waveDescriptor",
        "userText",
        "trigTimeArray",
        "waveArray1",
        "instrumentName",
        "instrumentNumber",
        "traceLabel",
        "waveArrayCount",
        "verticalGain",
        "verticalOffset",
        "nominalBits",
        "horizInterval",
        "horizOffset",
        "vertUnit",
        "horUnit",
        "sequenceSegments",
        "triggerTime",
        "recordType",
        "processingDone",
        "timeBase",
        "verticalCoupling",
        "bandwidthLimit",
        "waveSource"
    ],
)
