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

from lecroyparser.metadata import BinaryMetaDataStructure, AtomicType, MetaData
from lecroyparser.time_conversion import convert_time_stamp, convert_time_base


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


parse_uint8: partial[np.uint8] = partial(unpack, length=1, format_specifier="u1")
parse_uint16: partial[np.uint16] = partial(unpack, length=2, format_specifier="u2")
parse_int16: partial[np.int16] = partial(unpack, length=2, format_specifier="i2")
parse_int32: partial[np.int32] = partial(unpack, length=4, format_specifier="i4")
parse_float32: partial[np.float32] = partial(unpack, length=4, format_specifier="f4")
parse_float64: partial[np.float64] = partial(unpack, length=8, format_specifier="f8")
parse_bytes: partial[bytes] = partial(unpack, length=16, format_specifier="S16")


def parse_metadata( # pylint: disable=too-many-locals
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
    prs_bytes = partial(parse_bytes, data=data, offset=offset, endianness=endianness)

    # Add your parsing logic here
    binary_metadata: dict[str, int | float | str] = {}
    for name, prop in BinaryMetaDataStructure.items():
        if prop.atomic_type == AtomicType.undefined:
            binary_metadata[name] = "NOT PARSED"
        elif prop.atomic_type == AtomicType.uint8:
            binary_metadata[name] = int(prs_uint8(prop.location))
        elif prop.atomic_type == AtomicType.uint16:
            binary_metadata[name] = int(prs_uint16(prop.location))
        elif prop.atomic_type == AtomicType.int16:
            binary_metadata[name] = int(prs_int16(prop.location))
        elif prop.atomic_type == AtomicType.int32:
            binary_metadata[name] = int(prs_int32(prop.location))
        elif prop.atomic_type == AtomicType.float32:
            binary_metadata[name] = float(prs_float32(prop.location))
        elif prop.atomic_type == AtomicType.float64:
            binary_metadata[name] = float(prs_float64(prop.location))
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

    wave_source_list = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Unknown"]
    vertical_coupling_list = ["DC50", "GND", "DC1M", "GND", "AC1M"]
    bandwidth_limit_list = ["off", "on"]
    record_type_list = [
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
    processing_list = [
        "No Processing",
        "FIR Filter",
        "interpolated",
        "sparsed",
        "autoscaled",
        "no_resulst",
        "rolling",
        "cumulative",
    ]
    record_type = record_type_list[cast(int, binary_metadata["record_type"])]
    processing_done = processing_list[cast(int, binary_metadata["processing_done"])]
    time_base = convert_time_base(cast(int, binary_metadata["time_base"]))
    vertical_coupling = vertical_coupling_list[
        cast(int, binary_metadata["vertical_coupling"])
    ]
    bandwidth_limit = bandwidth_limit_list[
        cast(int, binary_metadata["bandwidth_limit"])
    ]
    wave_source = wave_source_list[cast(int, binary_metadata["wave_source"])]

    return MetaData(
        templateName=binary_metadata["template_name"],
        commType=binary_metadata["comm_type"],
        waveDescriptor=binary_metadata["wave_descriptor"],
        userText=binary_metadata["user_text"],
        trigTimeArray=binary_metadata["trig_time_array"],
        waveArray1=binary_metadata["wave_array1"],
        instrumentName=binary_metadata["instrument_name"],
        instrumentNumber=binary_metadata["instrument_number"],
        traceLabel=binary_metadata["trace_label"],
        waveArrayCount=binary_metadata["wave_array_count"],
        verticalGain=binary_metadata["vertical_gain"],
        verticalOffset=binary_metadata["vertical_offset"],
        nominalBits=binary_metadata["nominal_bits"],
        horizInterval=binary_metadata["horiz_interval"],
        horizOffset=binary_metadata["horiz_offset"],
        vertUnit=binary_metadata["vert_unit"],
        horUnit=binary_metadata["hor_unit"],
        sequenceSegments=binary_metadata["sequence_segments"],
        triggerTime=trigger_time,
        recordType=record_type,
        processingDone=processing_done,
        timeBase=time_base,
        verticalCoupling=vertical_coupling,
        bandwidthLimit=bandwidth_limit,
        waveSource=wave_source,
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
        + metadata.waveDescriptor
        + metadata.userText
        + metadata.trigTimeArray
    )
    if metadata.commType == 0:  # data is stored in 8bit integers
        y = np.frombuffer(
            data[start : start + metadata.waveArray1],
            dtype=np.dtype((endianness + "i1", metadata.waveArray1)),
            count=1,
        )[0]
    else:  # 16 bit integers
        length = metadata.waveArray1 // 2
        y = np.frombuffer(
            data[start : start + metadata.waveArray1],
            dtype=np.dtype((endianness + "i2", length)),
            count=1,
        )[0]

    # now scale the ADC values
    y = metadata.verticalGain * np.array(y) - metadata.verticalOffset

    x = (
        np.linspace(
            0,
            metadata.waveArrayCount * metadata.horizInterval,
            num=metadata.waveArrayCount,
        )
        + metadata.horizOffset
    )

    if sparse > 0:
        indices = int(len(x) / sparse) * np.arange(sparse)

        x = x[indices]
        y = y[indices]

    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((x, y))
    return combined_data, metadata
