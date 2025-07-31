"""
LeCroy Parser
This module provides functions to parse data from LeCroy oscilloscopes.
It includes functions to unpack data from bytes using different formats and endianness.
It also includes functions to parse specific data types such as int16, int32, float, double,
byte, word, and string.
"""

from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from lecroyparser.metadata import MetaData
from lecroyparser.time_conversion import convert_time_stamp, convert_time_base
from lecroyparser.binary_metadata_structure import AtomicType, BinaryMetaDataStructure 


WAVE_SOURCES = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Unknown"]
VERTICAL_COUPLINGS = ["DC50", "GND", "DC1M", "GND", "AC1M"]
BANDWIDTH_LIMITS = ["off", "on"]
RECORD_TYPES = [
    "single_sweep",
    "interleaved",
    "histogram",
    "graph",
    "filter_coefficient",
    "complex",
    "extrema",
    "sequence_obsolete",
    "centered_RIS",
    "peak_detect",
]
PROCESSING_TYPES = [
    "No Processing",
    "FIR Filter",
    "interpolated",
    "sparsed",
    "autoscaled",
    "no_resulst",
    "rolling",
    "cumulative",
]


def unpack(  # pylint: disable=too-many-arguments
    position: int,
    *,
    data: bytes,
    offset: int,
    length: int,
    endianness: str,
    format_specifier: str,
) -> Any:
    """
    Unpack the data from the given position and length.

    :param data: The data to unpack
    :param offset: The offset to start unpacking from
    :param position: The position to unpack from
    :param length: The length of the data to unpack
    :param endianness: The endianness of the data
    :param format_specifier: The format specifier for unpacking
    :return: The unpacked data
    """
    shifted_position = offset + position
    return np.frombuffer(
        data[shifted_position : shifted_position + length],
        f"{endianness}{format_specifier}",
        count=1,
    )[0]


"""
parse_uint8 is a partial function that unpacks a single byte as an unsigned integer.
"""
parse_uint8: partial[np.uint8] = partial(unpack, length=1, format_specifier="u1")

"""
parse_uint16 is a partial function that unpacks two bytes as an unsigned integer.
"""
parse_uint16: partial[np.uint16] = partial(unpack, length=2, format_specifier="u2")

"""
parse_int16 is a partial function that unpacks two bytes as a signed integer.
"""
parse_int16: partial[np.int16] = partial(unpack, length=2, format_specifier="i2")

"""
parse_int32 is a partial function that unpacks four bytes as a signed integer.
"""
parse_int32: partial[np.int32] = partial(unpack, length=4, format_specifier="i4")

"""
parse_float32 is a partial function that unpacks four bytes as a float.
"""
parse_float32: partial[np.float32] = partial(unpack, length=4, format_specifier="f4")

"""
parse_float64 is a partial function that unpacks eight bytes as a double.
"""
parse_float64: partial[np.float64] = partial(unpack, length=8, format_specifier="f8")

"""
parse_bytes is a partial function that unpacks 16 bytes as a string.
"""
parse_bytes: partial[bytes] = partial(unpack, length=16, format_specifier="S16", endianness="|")


def parse_metadata(  # pylint: disable=too-many-locals
    data: bytes, offset: int, endianness: str, second_digits: int = 3
) -> Any:
    """
    Parse the metadata from the data.

    :param data: The data to parse
    :param endianness: The endianness of the data
    :param second_digits: The number of digits after the decimal point for seconds
    :return: The parsed metadata
    """
    prs_uint16 = partial(parse_uint16, data=data, offset=offset, endianness=endianness)
    prs_int32 = partial(parse_int32, data=data, offset=offset, endianness=endianness)
    prs_float32 = partial(
        parse_float32, data=data, offset=offset, endianness=endianness
    )
    prs_float64 = partial(
        parse_float64, data=data, offset=offset, endianness=endianness
    )
    prs_uint8 = partial(parse_uint8, data=data, offset=offset, endianness=endianness)
    prs_int16 = partial(parse_int16, data=data, offset=offset, endianness=endianness)
    prs_bytes = partial(parse_bytes, data=data, offset=offset)

    # Add your parsing logic here
    binary_metadata: dict[
        str, np.uint8 | np.uint16 | np.int16 | np.int32 | np.float32 | np.float64 | str
    ] = {}
    for name, prop in BinaryMetaDataStructure.items():
        if prop.atomic_type == AtomicType.undefined:
            binary_metadata[name] = "NOT PARSED"
        elif prop.atomic_type == AtomicType.uint8:
            binary_metadata[name] = prs_uint8(prop.location)
        elif prop.atomic_type == AtomicType.uint16:
            binary_metadata[name] = prs_uint16(prop.location)
        elif prop.atomic_type == AtomicType.int16:
            binary_metadata[name] = prs_int16(prop.location)
        elif prop.atomic_type == AtomicType.int32:
            binary_metadata[name] = prs_int32(prop.location)
        elif prop.atomic_type == AtomicType.float32:
            binary_metadata[name] = prs_float32(prop.location)
        elif prop.atomic_type == AtomicType.float64:
            binary_metadata[name] = prs_float64(prop.location)
        elif prop.atomic_type == AtomicType.bytes:
            binary_metadata[name] = prs_bytes(prop.location).decode()
        else:
            raise ValueError(f"Unknown atomic type: {prop.atomic_type}")

    trigger_time = convert_time_stamp(
        cast(float, binary_metadata["trigger_seconds"]),
        cast(int, binary_metadata["trigger_minutes"]),
        cast(int, binary_metadata["trigger_hours"]),
        cast(int, binary_metadata["trigger_days"]),
        cast(int, binary_metadata["trigger_months"]),
        cast(int, binary_metadata["trigger_years"]),
        second_digits=second_digits,
    )

    record_type = RECORD_TYPES[cast(int, binary_metadata["record_type"])]
    processing_done = PROCESSING_TYPES[cast(int, binary_metadata["processing_done"])]
    time_base = convert_time_base(cast(int, binary_metadata["time_base"]))
    vertical_coupling = VERTICAL_COUPLINGS[
        cast(int, binary_metadata["vertical_coupling"])
    ]
    bandwidth_limit = BANDWIDTH_LIMITS[cast(int, binary_metadata["bandwidth_limit"])]
    wave_source = WAVE_SOURCES[cast(int, binary_metadata["wave_source"])]

    return MetaData(
        template_name=str(binary_metadata["template_name"]),
        comm_type=int(binary_metadata["comm_type"]),
        wave_descriptor=int(binary_metadata["wave_descriptor"]),
        user_text=int(binary_metadata["user_text"]),
        trig_time_array=int(binary_metadata["trig_time_array"]),
        wave_array1=int(binary_metadata["wave_array1"]),
        instrument_name=str(binary_metadata["instrument_name"]),
        instrument_number=int(binary_metadata["instrument_number"]),
        trace_label=str(binary_metadata["trace_label"]),
        wave_array_count=cast(np.int32, binary_metadata["wave_array_count"]),
        vertical_gain=cast(np.float32, binary_metadata["vertical_gain"]),
        vertical_offset=cast(np.float32, binary_metadata["vertical_offset"]),
        nominal_bits=int(binary_metadata["nominal_bits"]),
        horiz_interval=cast(np.float32, binary_metadata["horiz_interval"]),
        horiz_offset=cast(np.float64, binary_metadata["horiz_offset"]),
        vert_unit=str(binary_metadata["vert_unit"]),
        hor_unit=str(binary_metadata["hor_unit"]),
        sequence_segments=int(binary_metadata["sequence_segments"]),
        trigger_time=trigger_time,
        record_type=record_type,
        processing_done=processing_done,
        time_base=time_base,
        vertical_coupling=vertical_coupling,
        bandwidth_limit=bandwidth_limit,
        wave_source=wave_source,
    )


def parse_data(  # pylint: disable=too-many-locals, too-many-statements
    data: bytes, sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData]:
    """Parse the data."""
    # convert the first 50 bytes to a string to find position of substring WAVEDESC
    pos_wavedesc = data[:50].decode("ascii", "replace").index("WAVEDESC")

    # big endian (>) if 0, else little
    comm_order = partial(parse_uint16, data=data, offset=pos_wavedesc, endianness="<")(
        34
    )
    endianness = [">", "<"][comm_order]

    metadata = parse_metadata(
        data,
        offset=pos_wavedesc,
        endianness=endianness,
        second_digits=second_digits,
    )

    start = (
        pos_wavedesc
        + metadata.wave_descriptor
        + metadata.user_text
        + metadata.trig_time_array
    )
    if metadata.comm_type == 0:  # data is stored in 8bit integers
        y = np.frombuffer(
            data[start : start + metadata.wave_array1],
            dtype=np.dtype((endianness + "i1", metadata.wave_array1)),
            count=1,
        )[0]
    else:  # 16 bit integers
        length = metadata.wave_array1 // 2
        y = np.frombuffer(
            data[start : start + metadata.wave_array1],
            dtype=np.dtype((endianness + "i2", length)),
            count=1,
        )[0]

    # now scale the ADC values
    y = metadata.vertical_gain * np.array(y) - metadata.vertical_offset

    x = (
        np.linspace(
            0,
            metadata.wave_array_count * metadata.horiz_interval,
            num=metadata.wave_array_count,
        )
        + metadata.horiz_offset
    )

    if sparse > 0:
        indices = int(len(x) / sparse) * np.arange(sparse)

        x = x[indices]
        y = y[indices]

    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((x, y))
    return combined_data, metadata
