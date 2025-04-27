from enum import Enum
import functools
from collections import namedtuple
import sys
from typing import Optional
fromp pathlib import Path

import numpy as np

AtomicTypes = Enum("AtomicTypes", "INT16 INT32 FLOAT DOUBLE BYTE WORD STRING16")

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
        "waveSource",
    ],
)


def unpack(
    *,
    data: bytes,
    offset: int,
    position: int,
    length: int,
    endianness: str,
    format_specifier: str,
) -> np.ndarray:
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


def parse(
    position: int, *, atype: AtomicTypes, data: bytes, offset: int, endianness: str
) -> np.ndarray:
    """
    Parse the data at the given position for the given type.

    :param position: The position to parse from
    :param atype: The type of data to parse
    :param data: The data to parse
    :param offset: The offset to start parsing from
    :param endianness: The endianness of the data
    :return: The parsed data
    """
    parse_int16 = functools.partial(unpack, length=2, format_specifier="u2")
    parse_int32 = functools.partial(unpack, length=4, format_specifier="i4")
    parse_float = functools.partial(unpack, length=4, format_specifier="f4")
    parse_dble = functools.partial(unpack, length=8, format_specifier="f8")
    parse_byte = functools.partial(unpack, length=1, format_specifier="u1")
    parse_word = functools.partial(unpack, length=2, format_specifier="i2")
    parse_string = functools.partial(unpack, length=16, format_specifier="S16")
    match atype:
        case AtomicTypes.INT16:
            return parse_int16(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.INT32:
            return parse_int32(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.FLOAT:
            return parse_float(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.DOUBLE:
            return parse_dble(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.BYTE:
            return parse_byte(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.WORD:
            return parse_word(
                data=data, offset=offset, position=position, endianness=endianness
            )
        case AtomicTypes.STRING16:
            return parse_string(
                data=data, offset=offset, position=position, endianness=endianness
            )


def compose(f, g):
    """
    Compose two functions.

    :param f: The first function
    :param g: The second function
    :return: The composed function
    """
    return lambda x: f(g(x))


def convert_time_stamp(
    seconds: np.ndarray,
    minutes: np.ndarray,
    hours: np.ndarray,
    days: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    second_digits: int = 3,
) -> str:
    """
    Convert the time stamp to a human-readable format.

    :param seconds: The seconds
    :param minutes: The minutes
    :param hours: The hours
    :param days: The days
    :param months: The months
    :param years: The years
    :param second_digits: The number of digits after the decimal point for seconds
    :return: The formatted time stamp
    """
    second_format = "{:0" + str(second_digits + 3) + "." + str(second_digits) + "f}"
    full_format = "{}-{:02d}-{:02d} {:02d}:{:02d}:" + second_format

    return full_format.format(years, months, days, hours, minutes, seconds)


def convert_time_base(time_base_number: int) -> str:
    """Convert the time base number to a human-readable format.
    The time base number is an integer that encodes timing information as follows:
    0 : 1 ps  / div
    1:  2 ps / div
    2:  5 ps/div, up to 47 = 5 ks / div. 100 for external clock

    :param time_base_number: The time base number to convert
    :return: The human-readable time base
    """
    if time_base_number < 48:
        unit = "pnum k"[int(time_base_number / 9)]
        value = [1, 2, 5, 10, 20, 50, 100, 200, 500][time_base_number % 9]
        return "{} ".format(value) + unit.strip() + "s/div"
    elif time_base_number == 100:
        return "EXTERNAL"
    else:
        raise ValueError("Invalid time base number")


def parse_data(data: bytes, sparse=-1, secondDigits: int = 3) -> tuple[np.ndarray, MetaData]:
    """Parse the data."""
    waveSourceList = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Unknown"]
    verticalCouplingList = ["DC50", "GND", "DC1M", "GND", "AC1M"]
    bandwidthLimitList = ["off", "on"]
    recordTypeList = [
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
    processingList = [
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
    posWAVEDESC = data[:50].decode("ascii", "replace").index("WAVEDESC")

    commOrder = parse(
        34, atype=AtomicTypes.INT16, data=data, offset=posWAVEDESC, endianness="<"
    )  # big endian (>) if 0, else little
    endianness = [">", "<"][commOrder]
    prs_string = compose(
        bytes.decode,
        functools.partial(
            parse,
            atype=AtomicTypes.STRING16,
            data=data,
            offset=posWAVEDESC,
            endianness=endianness,
        ),
    )
    prs_int16 = functools.partial(
        parse,
        atype=AtomicTypes.INT16,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_int32 = functools.partial(
        parse,
        atype=AtomicTypes.INT32,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_float = functools.partial(
        parse,
        atype=AtomicTypes.FLOAT,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_dble = functools.partial(
        parse,
        atype=AtomicTypes.DOUBLE,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_byte = functools.partial(
        parse,
        atype=AtomicTypes.BYTE,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_word = functools.partial(
        parse,
        atype=AtomicTypes.WORD,
        data=data,
        offset=posWAVEDESC,
        endianness=endianness,
    )
    prs_time_base = compose(convert_time_base, prs_int16)

    templateName = prs_string(16)
    commType = prs_int16(32)  # encodes whether data is stored as 8 or 16bit

    waveDescriptor = prs_int32(36)
    userText = prs_int32(40)
    trigTimeArray = prs_int32(48)
    waveArray1 = prs_int32(60)

    instrumentName = prs_string(76)
    instrumentNumber = prs_int32(92)

    traceLabel = "NOT PARSED"
    waveArrayCount = prs_int32(116)

    verticalGain = prs_float(156)
    verticalOffset = prs_float(160)

    nominalBits = prs_int16(172)

    horizInterval = prs_float(176)
    horizOffset = prs_dble(180)

    vertUnit = "NOT PARSED"
    horUnit = "NOT PARSED"

    sequenceSegments = prs_int32(144)

    triggerSeconds = prs_dble(296)
    triggerMinutes = prs_byte(304)
    triggerHours = prs_byte(305)
    triggerDays = prs_byte(306)
    triggerMonths = prs_byte(307)
    triggerYears = prs_word(308)
    triggerTime = convert_time_stamp(
        triggerSeconds,
        triggerMinutes,
        triggerHours,
        triggerDays,
        triggerMonths,
        triggerYears,
        second_digits=secondDigits,
    )
    recordType = recordTypeList[prs_int16(316)]
    processingDone = processingList[prs_int16(318)]
    timeBase = prs_time_base(324)
    verticalCoupling = verticalCouplingList[prs_int16(326)]
    bandwidthLimit = bandwidthLimitList[prs_int16(334)]
    waveSource = waveSourceList[prs_int16(344)]

    start = posWAVEDESC + waveDescriptor + userText + trigTimeArray
    if commType == 0:  # data is stored in 8bit integers
        y = np.frombuffer(
            data[start : start + waveArray1],
            dtype=np.dtype((endianness + "i1", waveArray1)),
            count=1,
        )[0]
    else:  # 16 bit integers
        length = waveArray1 // 2
        y = np.frombuffer(
            data[start : start + waveArray1],
            dtype=np.dtype((endianness + "i2", length)),
            count=1,
        )[0]

    # now scale the ADC values
    y = verticalGain * np.array(y) - verticalOffset

    x = np.linspace(0, waveArrayCount * horizInterval, num=waveArrayCount) + horizOffset

    if sparse > 0:
        indices = int(len(x) / sparse) * np.arange(sparse)

        x = x[indices]
        y = y[indices]

    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((x, y))
    return combined_data, MetaData(
        templateName=templateName,
        commType=commType,
        waveDescriptor=waveDescriptor,
        userText=userText,
        trigTimeArray=trigTimeArray,
        waveArray1=waveArray1,
        instrumentName=instrumentName,
        instrumentNumber=instrumentNumber,
        traceLabel=traceLabel,
        waveArrayCount=waveArrayCount,
        verticalGain=verticalGain,
        verticalOffset=verticalOffset,
        nominalBits=nominalBits,
        horizInterval=horizInterval,
        horizOffset=horizOffset,
        vertUnit=vertUnit,
        horUnit=horUnit,
        sequenceSegments=sequenceSegments,
        triggerTime=triggerTime,
        recordType=recordType,
        processingDone=processingDone,
        timeBase=timeBase,
        verticalCoupling=verticalCoupling,
        bandwidthLimit=bandwidthLimit,
        waveSource=waveSource,
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
    filenames: list[str], sparse=-1, secondDigits: int = 3
) -> tuple[np.ndarray, MetaData]:
    """
    Parse the data from multiple leCroy binary waveform files.

    :param filenames: List of paths to the leCroy binary waveform files
    :param sparse: Number of points to skip in the x and y data
    :param secondDigits: Number of digits after the decimal point for seconds
    :return: Tuple of data and metadata
    """
    data = np.ndarray([])
    meta = None
    for filename in filenames:
        assert filename.endswith(".trc")
        data_temp, meta_temp = parse_data_from_file(
            filename, sparse=sparse, secondDigits=secondDigits
        )
        data = np.column_stack((data, data_temp[:, 1])) if data.size else data_temp
        if meta is None:
            meta = meta_temp
    return data, meta


def parse_data_from_file(filename: str, sparse=-1, secondDigits: int = 3) -> tuple[np.ndarray, MetaData]:
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
    return parse_data(content, sparse=sparse, secondDigits=secondDigits)


def dump(
    data: np.ndarray,
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
    writer = functools.partial(np.savetxt, X=data, header=str(metadata), fmt="%+15.12e")
    if output_filename:
        writer(output_filename)
    else:
        writer(sys.stdout)


def convert_to_text_file(filename: str) -> None:
    """
    Convert a leCroy binary waveform file to a text file.
    The text file will contain the x and y data in a formatted manner.
    The output file will have the same name as the input file, but with a .dat extension.

    :param filename: Path to the leCroy binary waveform file
    :return: None
    """
    data, meta = parse_data_from_file(filename)
    output_name = filename.replace(".trc", ".dat")
    dump(data, metadata=meta, output_filename=output_name)
