"""
Test the unpack function from parsing.py
"""
# mypy: ignore-errors
import struct

import numpy as np
from lecroyparser.parsing import unpack


def test_unpack_null_offset_null_position_uint16():
    """
    Test the unpack function from scope_data_func.py

    Offset and position are both 0
    The input data is a 2-byte unsigned integer
    The expected output is the integer value of the input data

    The test checks both big-endian and little-endian formats
    """
    offset = 0
    position = 0

    # Generate a random 2-byte unsigned integer
    # The range is from 0 to 65535 (2^16 - 1)
    # because we are using 2 bytes
    min_value = np.iinfo(np.uint16).min
    max_value = np.iinfo(np.uint16).max
    # assert uint16_min_value == 0
    expected_output = np.random.randint(min_value, max_value)
    input_data = expected_output.to_bytes(2, byteorder="big") # pylint: disable=no-member

    # Test Big Endian
    result = unpack(data=input_data,
                   offset=offset,
                   position=position,
                   length=2,
                   endianness=">",
                   format_specifier="u2")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    #  Test Little Endian
    result = unpack(data=input_data[::-1],
                   offset=offset,
                   position=position,
                   length=2,
                   endianness="<",
                   format_specifier="u2")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_unpack_null_offset_null_position_str():
    """
    Test the unpack function from scope_data_func.py

    Offset and position are both 0
    The input data is a 6-byte string
    The expected output is the string value of the input data

    The test checks that endianness does not affect the string unpacking, because
    strings are not affected by endianness
    """
    offset = 0
    position = 0

    expected_output = "Lecroy"
    input_data = bytes(expected_output, "utf-8")

    result = unpack(data=input_data,
                   offset=offset,
                   position=position,
                   length=6,
                   endianness=">",
                   format_specifier="S6")
    assert result.decode() == expected_output, f"Expected {expected_output}, but got {result}"

    # Changing the endianness should not affect the result
    result = unpack(data=input_data,
                   offset=offset,
                   position=position,
                   length=6,
                   endianness="<",
                   format_specifier="S6")
    assert result.decode() == expected_output, f"Expected {expected_output}, but got {result}"


def test_unpack_null_offset_null_position_float32():
    """
    Test the unpack function from scope_data_func.py

    Offset and position are both 0
    The input data is a 4-byte float
    The expected output is the float value of the input data

    The test checks both big-endian and little-endian formats
    """
    offset = 0
    position = 0

    min_value = np.finfo(np.float32).min
    max_value = np.finfo(np.float32).max
    expected_output = np.random.uniform(min_value, max_value)
    input_data = struct.pack(">f", expected_output)

    # Test Big Endian
    result = unpack(data=input_data,
                   offset=offset,
                   position=position,
                   length=4,
                   endianness=">",
                   format_specifier="f4")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    #  Test Little Endian
    result = unpack(data=input_data[::-1],
                   offset=offset,
                   position=position,
                   length=4,
                   endianness="<",
                   format_specifier="f4")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
