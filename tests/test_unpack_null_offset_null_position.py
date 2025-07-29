"""
Test the unpack function from parsing.py

Offset and position are both 0
"""
# mypy: ignore-errors
import struct

import numpy as np
from pytest import mark
from lecroyparser.parsing import unpack


OFFSET = 0
POSITION = 0

@mark.parametrize("nptype,atomic_size,format_specifier", [
    (np.uint8, 1, "u1"),
    (np.uint16, 2, "u2"),
    (np.int16, 2, "i2"),
    (np.int32, 4, "i4"),
])
def test_unpack(nptype, atomic_size, format_specifier):
    """
    Test the unpack function from parsing.py for various data types

    The input data is a byte array of the specified size
    The expected output is the integer value of the input data

    The test checks both big-endian and little-endian formats
    """
    # Generate a random integer within the range of the specified numpy type
    min_value = np.iinfo(nptype).min
    max_value = np.iinfo(nptype).max
    expected_output = np.random.randint(min_value, max_value + 1)
    if nptype.__name__.startswith("u"):
        input_data = expected_output.to_bytes(atomic_size, byteorder="big")
    else:
        input_data = expected_output.to_bytes(atomic_size, byteorder="big", signed=True)
    # Test Big Endian
    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=atomic_size,
                   endianness=">",
                   format_specifier=format_specifier)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test Little Endian (input data is reversed)
    result = unpack(data=input_data[::-1],
                   offset=OFFSET,
                   position=POSITION,
                   length=atomic_size,
                   endianness="<",
                   format_specifier=format_specifier)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Test out of range values failure
    min_value = np.iinfo(nptype).max + 1
    max_value = np.iinfo(nptype).max + 10
    expected_output = np.random.randint(min_value, max_value)
    if nptype.__name__.startswith("u"):
        input_data = expected_output.to_bytes(atomic_size * 2, byteorder="big")
    else:
        input_data = expected_output.to_bytes(atomic_size * 2, byteorder="big", signed=True)
    # Test Big Endian
    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=atomic_size,
                   endianness=">",
                   format_specifier=format_specifier)
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
