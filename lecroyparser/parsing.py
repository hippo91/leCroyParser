"""
LeCroy Parser
This module provides functions to parse data from LeCroy oscilloscopes.
It includes functions to unpack data from bytes using different formats and endianness.
It also includes functions to parse specific data types such as int16, int32, float, double, 
byte, word, and string.
"""
from typing import Any
from functools import partial

import numpy as np


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
