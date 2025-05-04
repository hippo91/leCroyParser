"""
lecroyparser.scope_data_fun

Reimplements initial ScopeData class from the original lecroyparser
but with a functional approach.
Introduces type hints and uses numpy for data handling.
"""

from functools import partial
import sys
from typing import Any, Optional, Callable, cast
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lecroyparser.parsing import (
    parse_uint16,
    parse_int32,
    parse_float32,
    parse_float64,
    parse_uint8,
    parse_int16,
    parse_bytes,
)
from lecroyparser.metadata import BinaryMetaDataStructure, AtomicType, MetaData
from lecroyparser.time_conversion import convert_time_stamp, convert_time_base


def compose(f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose two functions.

    :param f: The first function
    :param g: The second function
    :return: The composed function
    """
    return lambda x: f(g(x))

def parse_metadata(data: bytes, offset: int, endianness: str, second_digits: int = 3) -> Any:
    """
    Parse the metadata from the data.

    :param data: The data to parse
    :param endianness: The endianness of the data
    :param second_digits: The number of digits after the decimal point for seconds
    :return: The parsed metadata
    """
    prs_uint16 = partial(parse_uint16, data=data, offset=offset, endianness=endianness)
    prs_int32 = partial(parse_int32, data=data, offset=offset, endianness=endianness)
    prs_float32 = partial(parse_float32, data=data, offset=offset, endianness=endianness)
    prs_float64 = partial(parse_float64, data=data, offset=offset, endianness=endianness)
    prs_uint8 = partial(parse_uint8, data=data, offset=offset, endianness=endianness)
    prs_int16 = partial(parse_int16, data=data, offset=offset, endianness=endianness)
    prs_bytes = partial(parse_bytes, data=data, offset=offset, endianness=endianness)

    # Add your parsing logic here
    binary_metadata = {}
    for name, prop in BinaryMetaDataStructure.items():
        if prop == "NOT PARSED":
            value = prop
        elif prop.atomic_type == AtomicType.uint8:
            value = prs_uint8(prop.location)
        elif prop.atomic_type == AtomicType.uint16:
            value = prs_uint16(prop.location)
        elif prop.atomic_type == AtomicType.int16:
            value = prs_int16(prop.location)
        elif prop.atomic_type == AtomicType.int32:
            value = prs_int32(prop.location)
        elif prop.atomic_type == AtomicType.float32:
            value = prs_float32(prop.location)
        elif prop.atomic_type == AtomicType.float64:
            value = prs_float64(prop.location)
        elif prop.atomic_type == AtomicType.bytes:
            value = prs_bytes(prop.location)
        else:
            raise ValueError(f"Unknown atomic type: {prop.atomic_type}")
        binary_metadata[name] = value

    trigger_time = convert_time_stamp(
        binary_metadata["trigger_seconds"],
        binary_metadata["trigger_minutes"],
        binary_metadata["trigger_hours"],
        binary_metadata["trigger_days"],
        binary_metadata["trigger_months"],
        binary_metadata["trigger_years"],
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
    record_type = record_type_list[binary_metadata["record_type"]]
    processing_done = processing_list[binary_metadata["processing_done"]]
    time_base = convert_time_base(binary_metadata["time_base"])
    vertical_coupling = vertical_coupling_list[binary_metadata["vertical_coupling"]]
    bandwidth_limit = bandwidth_limit_list[binary_metadata["bandwidth_limit"]]
    wave_source = wave_source_list[binary_metadata["wave_source"]]

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

    start = pos_wavedesc + metadata.waveDescriptor + metadata.userText + metadata.trigTimeArray
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
        np.linspace(0, metadata.waveArrayCount * metadata.horizInterval, num=metadata.waveArrayCount)
        + metadata.horizOffset
    )

    if sparse > 0:
        indices = int(len(x) / sparse) * np.arange(sparse)

        x = x[indices]
        y = y[indices]

    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((x, y))
    return combined_data, metadata


def find_channels_files(first_channel_file_path: str) -> list[str]:
    """
    Find all leCroy binary waveform files in the given directory.

    :param first_channel_file_path: Path to the first channel file
    :return: List of paths to the leCroy binary waveform files
    """
    filepath = Path(first_channel_file_path)
    dir_path = filepath.parent
    filename = filepath.name
    if not filename.endswith(".trc"):
        raise ValueError("The file must be a .trc file")
    if not filename.startswith("C"):
        raise ValueError("The file must start with C")
    if not dir_path.is_dir():
        raise ValueError("The path must be a directory")
    channel_common_part = filename[2:]
    return [str(_path) for _path in dir_path.glob(f"C[0-9]{channel_common_part}")]


def parse_data_from_multiple_files(
    filenames: list[str], sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData | None]:
    """
    Parse the data from multiple leCroy binary waveform files.

    :param filenames: List of paths to the leCroy binary waveform files
    :param sparse: Number of points to skip in the x and y data
    :param secondDigits: Number of digits after the decimal point for seconds
    :return: Tuple of data and metadata
    """
    data: npt.NDArray[np.floating[Any]] = np.ndarray([])
    meta = None
    for filename in filenames:
        assert filename.endswith(".trc")
        data_temp, meta_temp = parse_data_from_file(
            filename, sparse=sparse, second_digits=second_digits
        )
        data = np.column_stack((data, data_temp[:, 1])) if data.size else data_temp
        if meta is None:
            meta = meta_temp
    return data, meta


def parse_data_from_file(
    filename: str, sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData]:
    """
    Parse the data from a leCroy binary waveform file.

    :param filename: Path to the leCroy binary waveform file
    :param sparse: Number of points to skip in the x and y data
    :param secondDigits: Number of digits after the decimal point for seconds
    :return: Tuple of data and metadata
    """
    assert filename.endswith(".trc")
    with open(filename, "rb") as f:
        content = f.read()
    return parse_data(content, sparse=sparse, second_digits=second_digits)


def dump(
    data: npt.NDArray[np.floating[Any]],
    metadata: Optional[MetaData] = None,
    output_filename: Optional[str] = None,
) -> None:
    """
    Dump the content of the data object to the console or to the file in argument if any.
    This function prints the x and y data in a formatted manner.
    It checks that the x and y data are numpy arrays of the same shape.

    :param data: object containing the waveform data
    :return: None
    """
    assert data.shape[1] == 2
    writer = partial(np.savetxt, X=data, header=str(metadata), fmt="%+15.12e")
    if output_filename:
        writer(output_filename)
    else:
        writer(sys.stdout)


def convert_to_text_file(
    filename: str, sparse: int = -1, second_digits: int = 3, parse_all: bool = False
) -> None:
    """
    Convert a leCroy binary waveform file to a text file.
    The text file will contain the x and y data in a formatted manner.
    The output file will have the same name as the input file, but with a .dat extension.

    :param filename: Path to the leCroy binary waveform file
    :return: None
    """
    if parse_all:
        filenames = find_channels_files(filename)
        data, meta = parse_data_from_multiple_files(
            filenames, sparse=sparse, second_digits=second_digits
        )
    else:
        data, meta = parse_data_from_file(
            filename, sparse=sparse, second_digits=second_digits
        )
    output_name = filename.replace(".trc", ".dat")
    dump(data, metadata=meta, output_filename=output_name)
