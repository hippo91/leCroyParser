"""
Test the unpack function from parsing.py

Offset and position are both 0
"""
# mypy: ignore-errors
import struct

import numpy as np
from lecroyparser.parsing import unpack


OFFSET = 0
POSITION = 0

def test_uint8():
    """
    Test the unpack function from scope_data_func.py for uint8

    The input data is a single byte
    The expected output is the integer value of the input data

    The test checks both big-endian and little-endian formats
    """
    # Generate a random unsigned integer between 0 and 255
    # The range is from 0 to 255 (2^8 - 1)
    # because we are using 1 byte
    min_value = np.iinfo(np.uint8).min
    max_value = np.iinfo(np.uint8).max
    # assert uint8_min_value == 0
    expected_output = np.random.randint(min_value, max_value + 1)
    input_data = expected_output.to_bytes(1, byteorder="big") # pylint: disable=no-member

    # Test Big Endian
    result = unpack(data=input_data,
                offset=OFFSET,
                position=POSITION,
                length=1,
                endianness=">",
                format_specifier="u1")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    #  Test Little Endian (input data is reversed)
    result = unpack(data=input_data[::-1],
                offset=OFFSET,
                position=POSITION,
                length=1,
                endianness="<",
                format_specifier="u1")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test out of range values failure
    min_value = np.iinfo(np.uint8).max
    max_value = np.iinfo(np.uint16).max
    # assert uint8_min_value == 0
    expected_output = np.random.randint(min_value, max_value)
    input_data = expected_output.to_bytes(2, byteorder="big") # pylint: disable=no-member
    # Test Big Endian
    result = unpack(data=input_data,
                offset=OFFSET,
                position=POSITION,
                length=1,
                endianness=">",
                format_specifier="u1")
    assert result != expected_output, f"Expected {expected_output}, but got {result}"

def test_uint16():
    """
    Test the unpack function from scope_data_func.py for uint16

    The input data is a 2-byte unsigned integer
    The expected output is the integer value of the input data

    The test checks both big-endian and little-endian formats
    """
    # Generate a random 2-byte unsigned integer
    # The range is from 0 to 65535 (2^16 - 1)
    # because we are using 2 bytes
    min_value = np.iinfo(np.uint16).min
    max_value = np.iinfo(np.uint16).max
    # assert uint16_min_value == 0
    expected_output = np.random.randint(min_value, max_value + 1)
    input_data = expected_output.to_bytes(2, byteorder="big") # pylint: disable=no-member

    # Test Big Endian
    result = unpack(data=input_data,
                offset=OFFSET,
                position=POSITION,
                length=2,
                endianness=">",
                format_specifier="u2")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    #  Test Little Endian (input data is reversed)
    result = unpack(data=input_data[::-1],
                offset=OFFSET,
                position=POSITION,
                length=2,
                endianness="<",
                format_specifier="u2")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test out of range values failure
    min_value = np.iinfo(np.uint16).max
    max_value = np.iinfo(np.uint32).max
    # assert uint16_min_value == 0
    expected_output = np.random.randint(min_value, max_value)
    input_data = expected_output.to_bytes(4, byteorder="big") # pylint: disable=no-member

    # Test Big Endian
    result = unpack(data=input_data,
                offset=OFFSET,
                position=POSITION,
                length=2,
                endianness=">",
                format_specifier="u2")
    assert result != expected_output, f"Expected {expected_output}, but got {result}"


def test_str():
    """
    Test the unpack function from scope_data_func.py for string

    The input data is a 6-byte string
    The expected output is the string value of the input data

    The test checks that endianness does not affect the string unpacking, because
    strings are not affected by endianness
    """
    expected_output = "Lecroy"
    input_data = bytes(expected_output, "utf-8")

    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=6,
                   endianness=">",
                   format_specifier="S6")
    assert result.decode() == expected_output, f"Expected {expected_output}, but got {result}"

    # Changing the endianness should not affect the result
    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=6,
                   endianness="<",
                   format_specifier="S6")
    assert result.decode() == expected_output, f"Expected {expected_output}, but got {result}"


def test_float32():
    """
    Test the unpack function from scope_data_func.py for float32

    The input data is a 4-byte float
    The expected output is the float value of the input data

    The test checks both big-endian and little-endian formats
    """
    min_value = np.finfo(np.float32).min
    max_value = np.finfo(np.float32).max
    expected_output = np.random.uniform(min_value, max_value)
    input_data = struct.pack(">f", expected_output)

    # Test Big Endian
    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=4,
                   endianness=">",
                   format_specifier="f4")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    #  Test Little Endian
    result = unpack(data=input_data[::-1],
                   offset=OFFSET,
                   position=POSITION,
                   length=4,
                   endianness="<",
                   format_specifier="f4")
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
