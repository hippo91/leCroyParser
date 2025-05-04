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
    parse_int16,
    parse_int32,
    parse_float,
    parse_dble,
    parse_byte,
    parse_word,
    parse_string,
)
from lecroyparser.metadata import MetaData
from lecroyparser.time_conversion import convert_time_stamp, convert_time_base


def compose(f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose two functions.

    :param f: The first function
    :param g: The second function
    :return: The composed function
    """
    return lambda x: f(g(x))


def parse_data(  # pylint: disable=too-many-locals, too-many-statements
    data: bytes, sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData]:
    """Parse the data."""
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

    # convert the first 50 bytes to a string to find position of substring WAVEDESC
    pos_wavedesc = data[:50].decode("ascii", "replace").index("WAVEDESC")

    # big endian (>) if 0, else little
    comm_order = partial(parse_int16, data=data, offset=pos_wavedesc, endianness="<")(
        34
    )
    endianness = [">", "<"][comm_order]
    prs_string = partial(
        parse_string, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_int16 = partial(
        parse_int16, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_index = compose(partial(cast, int), prs_int16)
    prs_int32 = partial(
        parse_int32, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_float = partial(
        parse_float, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_dble = partial(
        parse_dble, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_byte = partial(
        parse_byte, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_word = partial(
        parse_word, data=data, offset=pos_wavedesc, endianness=endianness
    )
    prs_time_base = compose(convert_time_base, prs_int16)

    template_name = prs_string(16)
    comm_type = prs_int16(32)  # encodes whether data is stored as 8 or 16bit

    wave_descriptor = prs_int32(36)
    user_text = prs_int32(40)
    trig_time_array = prs_int32(48)
    wave_array1 = prs_int32(60)

    instrument_name = prs_string(76)
    instrument_number = prs_int32(92)

    trace_label = "NOT PARSED"
    wave_array_count = prs_int32(116)

    vertical_gain = prs_float(156)
    vertical_offset = prs_float(160)

    nominal_bits = prs_int16(172)

    horiz_interval = prs_float(176)
    horiz_offset = prs_dble(180)

    vert_unit = "NOT PARSED"
    hor_unit = "NOT PARSED"

    sequence_segments = prs_int32(144)

    trigger_seconds = prs_dble(296)
    trigger_minutes = prs_byte(304)
    trigger_hours = prs_byte(305)
    trigger_days = prs_byte(306)
    trigger_months = prs_byte(307)
    trigger_years = prs_word(308)
    trigger_time = convert_time_stamp(
        trigger_seconds,
        trigger_minutes,
        trigger_hours,
        trigger_days,
        trigger_months,
        trigger_years,
        second_digits=second_digits,
    )
    record_type = record_type_list[prs_index(316)]
    processing_done = processing_list[prs_index(318)]
    time_base = prs_time_base(324)
    vertical_coupling = vertical_coupling_list[prs_index(326)]
    bandwidth_limit = bandwidth_limit_list[prs_index(334)]
    wave_source = wave_source_list[prs_index(344)]

    start = pos_wavedesc + wave_descriptor + user_text + trig_time_array
    if comm_type == 0:  # data is stored in 8bit integers
        y = np.frombuffer(
            data[start : start + wave_array1],
            dtype=np.dtype((endianness + "i1", wave_array1)),
            count=1,
        )[0]
    else:  # 16 bit integers
        length = wave_array1 // 2
        y = np.frombuffer(
            data[start : start + wave_array1],
            dtype=np.dtype((endianness + "i2", length)),
            count=1,
        )[0]

    # now scale the ADC values
    y = vertical_gain * np.array(y) - vertical_offset

    x = (
        np.linspace(0, wave_array_count * horiz_interval, num=wave_array_count)
        + horiz_offset
    )

    if sparse > 0:
        indices = int(len(x) / sparse) * np.arange(sparse)

        x = x[indices]
        y = y[indices]

    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((x, y))
    return combined_data, MetaData(
        templateName=template_name,
        commType=comm_type,
        waveDescriptor=wave_descriptor,
        userText=user_text,
        trigTimeArray=trig_time_array,
        waveArray1=wave_array1,
        instrumentName=instrument_name,
        instrumentNumber=instrument_number,
        traceLabel=trace_label,
        waveArrayCount=wave_array_count,
        verticalGain=vertical_gain,
        verticalOffset=vertical_offset,
        nominalBits=nominal_bits,
        horizInterval=horiz_interval,
        horizOffset=horiz_offset,
        vertUnit=vert_unit,
        horUnit=hor_unit,
        sequenceSegments=sequence_segments,
        triggerTime=trigger_time,
        recordType=record_type,
        processingDone=processing_done,
        timeBase=time_base,
        verticalCoupling=vertical_coupling,
        bandwidthLimit=bandwidth_limit,
        waveSource=wave_source,
    )


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
