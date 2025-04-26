from enum import Enum
import functools
from collections import namedtuple
import sys
from typing import Optional

import numpy as np

AtomicTypes = Enum("AtomicTypes", "INT16 INT32 FLOAT DOUBLE BYTE WORD")

MetaData = namedtuple("MetaData", [
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
])


def unpack(*, data: bytes, offset: int, position: int, length: int, endianness: str, format_specifier: str):
    """a wrapper that reads binary data
    in a given position in the file, with correct endianness, and returns the parsed
    data as a tuple, according to the format specifier."""
    shifted_position = offset + position
    return np.frombuffer(data[shifted_position : shifted_position + length], f"{endianness}{format_specifier}", count=1)[0]


def parse(position: int, * , atype: AtomicTypes, data: bytes, offset: int,  endianness:str):
    parse_int16 = functools.partial(unpack, length=2, format_specifier="u2")
    parse_int32 = functools.partial(unpack, length=4, format_specifier="i4")
    parse_float = functools.partial(unpack, length=4, format_specifier="f4")
    parse_dble = functools.partial(unpack, length=8, format_specifier="f8")
    parse_byte = functools.partial(unpack, length=1, format_specifier="u1")
    parse_word = functools.partial(unpack, length=2, format_specifier="i2")
    match atype:
        case AtomicTypes.INT16:
            return parse_int16(data=data, offset=offset, position=position, endianness=endianness)
        case AtomicTypes.INT32:
            return parse_int32(data=data, offset=offset, position=position, endianness=endianness)
        case AtomicTypes.FLOAT:
            return parse_float(data=data, offset=offset, position=position, endianness=endianness)
        case AtomicTypes.DOUBLE:
            return parse_dble(data=data, offset=offset, position=position, endianness=endianness)
        case AtomicTypes.BYTE:
            return parse_byte(data=data, offset=offset, position=position, endianness=endianness)
        case AtomicTypes.WORD:
            return parse_word(data=data, offset=offset, position=position, endianness=endianness)


def parse_string(
    position: int,
    *,
    data: bytes,
    offset: int,
    endianness: str,
    length: int = 16,
):
    s = unpack(data=data, offset=offset, position=position, length=length, endianness=endianness, format_specifier=f"S{length}")
    if sys.version_info > (3, 0):
        s = s.decode("ascii")
    return s


def parse_time_stamp(
    position: int,
    *,
    data: bytes,
    offset: int,
    endianness: str,
    second_digits: int = 3):

    prs_dble = functools.partial(parse, atype=AtomicTypes.DOUBLE, data=data, offset=offset, endianness=endianness)
    prs_byte = functools.partial(parse, atype=AtomicTypes.BYTE, data=data, offset=offset, endianness=endianness)
    prs_word = functools.partial(parse, atype=AtomicTypes.WORD, data=data, offset=offset, endianness=endianness)

    second = prs_dble(position)
    minute = prs_byte(position + 8)
    hour = prs_byte(position + 9)
    day = prs_byte(position + 10)
    month = prs_byte(position + 11)
    year = prs_word(position + 12)

    second_format = "{:0" + str(second_digits + 3) + "." + str(second_digits) + "f}"
    full_format = "{}-{:02d}-{:02d} {:02d}:{:02d}:" + second_format

    return full_format.format(year, month, day, hour, minute, second)


def parse_time_base(
    pos: int,
    *,
    data: bytes,
    offset: int,
    endianness: str
):
    """time base is an integer, and encodes timing information as follows:
    0 : 1 ps  / div
    1:  2 ps / div
    2:  5 ps/div, up to 47 = 5 ks / div. 100 for external clock"""

    time_base_number = parse(pos, atype=AtomicTypes.INT16, data=data, offset=offset, endianness=endianness)

    if time_base_number < 48:
        unit = "pnum k"[int(time_base_number / 9)]
        value = [1, 2, 5, 10, 20, 50, 100, 200, 500][time_base_number % 9]
        return "{} ".format(value) + unit.strip() + "s/div"
    elif time_base_number == 100:
        return "EXTERNAL"
    

def parse_data(data: bytes, sparse=-1, secondDigits: int = 3):
    """Parse the data."""
    waveSourceList = ["Channel 1", "Channel 2", "Channel 3", "Channel 4", "Unknown"]
    verticalCouplingList = ["DC50", "GND", "DC1M", "GND", "AC1M"]
    bandwidthLimitList = ["off", "on"]
    recordTypeList = ["single_sweep", "interleaved", "histogram", "graph",
                        "filter_coefficient", "complex", "extrema", "sequence_obsolete",
                        "centered_RIS", "peak_detect"]
    processingList = ["No Processing", "FIR Filter", "interpolated", "sparsed",
                        "autoscaled", "no_resulst", "rolling", "cumulative"]

    # convert the first 50 bytes to a string to find position of substring WAVEDESC
    posWAVEDESC = data[:50].decode("ascii", "replace").index("WAVEDESC")
    
    commOrder = parse(34, atype=AtomicTypes.INT16, data=data, offset=posWAVEDESC, endianness="<")  # big endian (>) if 0, else little
    endianness = [">", "<"][commOrder]
    prs_string = functools.partial(parse_string, data=data, offset=posWAVEDESC, endianness=endianness)
    prs_int16 = functools.partial(parse, atype=AtomicTypes.INT16, data=data, offset=posWAVEDESC, endianness=endianness)
    prs_int32 = functools.partial(parse, atype=AtomicTypes.INT32, data=data, offset=posWAVEDESC, endianness=endianness)
    prs_float = functools.partial(parse, atype=AtomicTypes.FLOAT, data=data, offset=posWAVEDESC, endianness=endianness)
    prs_dble = functools.partial(parse, atype=AtomicTypes.DOUBLE, data=data, offset=posWAVEDESC, endianness=endianness)
    prs_time_stamp = functools.partial(parse_time_stamp, data=data, offset=posWAVEDESC, endianness=endianness, second_digits=secondDigits)
    prs_time_base = functools.partial(parse_time_base, data=data, offset=posWAVEDESC, endianness=endianness)

    templateName = prs_string(16)
    commType = prs_int16(32)  # encodes whether data is stored as 8 or 16bit

    waveDescriptor = prs_int32(36)
    userText = prs_int32(40)
    trigTimeArray = prs_int32(48)
    waveArray1 = prs_int32(60)

    instrumentName = prs_string(position=76)
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

    triggerTime = prs_time_stamp(296)
    recordType = recordTypeList[prs_int16(316)]
    processingDone = processingList[prs_int16(318)]
    timeBase = prs_time_base(324)
    verticalCoupling = verticalCouplingList[prs_int16(326)]
    bandwidthLimit = bandwidthLimitList[prs_int16(334)]
    waveSource = waveSourceList[prs_int16(344)]

    start = posWAVEDESC + waveDescriptor + userText + trigTimeArray
    if commType == 0:  # data is stored in 8bit integers
        y = np.frombuffer(data[start:start + waveArray1], dtype=np.dtype((endianness + "i1", waveArray1)), count=1)[0]
    else:  # 16 bit integers
        length = waveArray1 // 2
        y = np.frombuffer(data[start:start + waveArray1], dtype=np.dtype((endianness + "i2", length)), count=1)[0]

    # now scale the ADC values
    y = verticalGain * np.array(y) - verticalOffset

    x = np.linspace(0, waveArrayCount * horizInterval,
                    num=waveArrayCount) + horizOffset

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


def dump(data: np.ndarray, metadata: Optional[MetaData] = None, output_filename: Optional[str] = None):
    """
    Dump the content of the data object to the console or to the file in argument if any.
    This function prints the x and y data in a formatted manner.
    It checks that the x and y data are numpy arrays of the same shape.

    :param data: object containing the waveform data
    :return: None
    """
    assert(data.shape[1] == 2)
    writer = functools.partial(np.savetxt, X=data, header=str(metadata), fmt="%+15.12e")
    if output_filename:
        writer(output_filename)
    else:
        writer(sys.stdout)


def convert_to_text_file(filename: str):
    """
    Convert a leCroy binary waveform file to a text file.
    The text file will contain the x and y data in a formatted manner.
    The output file will have the same name as the input file, but with a .dat extension.

    :param filename: Path to the leCroy binary waveform file
    :return: None
    """
    assert(filename.endswith(".trc"))
    with open(filename, "rb") as f:
        content = f.read()
    data, meta = parse_data(content)
    output_name = filename.replace(".trc", ".dat")
    dump(data, metadata=meta, output_filename=output_name)