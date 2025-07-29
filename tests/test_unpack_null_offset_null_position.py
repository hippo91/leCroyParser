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


def _big_endian_test_helper(input_data, expected_output, atomic_size, format_specifier):
    """
    Helper function to test unpacking with big-endian data
    """
    result = unpack(data=input_data,
                   offset=OFFSET,
                   position=POSITION,
                   length=atomic_size,
                   endianness=">",
                   format_specifier=format_specifier)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def _little_endian_test_helper(input_data, expected_output, atomic_size, format_specifier):
    """
    Helper function to test unpacking with little-endian data
    """
    # Reversing the input data for little-endian test
    result = unpack(data=input_data[::-1],
                   offset=OFFSET,
                   position=POSITION,
                   length=atomic_size,
                   endianness="<",
                   format_specifier=format_specifier)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


@mark.parametrize("nptype,atomic_size,format_specifier", [
    (np.uint8, 1, "u1"),
    (np.uint16, 2, "u2"),
    (np.int16, 2, "i2"),
    (np.int32, 4, "i4"),
])
def test_unpack_integer(nptype, atomic_size, format_specifier):
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

    _big_endian_test_helper(input_data, expected_output, atomic_size, format_specifier)
    _little_endian_test_helper(input_data, expected_output, atomic_size, format_specifier)

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


@mark.parametrize("nptype,atomic_size,format_specifier", [
    (np.float32, 4, "f4"),
    (np.float64, 8, "f8"),
])
def test_unpack_float(nptype, atomic_size, format_specifier):
    """
    Test the unpack function from parsing.py for float data type

    The input data is a byte array of the specified size
    The expected output is the float value of the input data

    The test checks both big-endian and little-endian formats
    """
    min_value = np.finfo(nptype).min / 2.
    max_value = np.finfo(nptype).max / 2.
    # Halving min and max values to avoid overflow in uniform generation
    # See https://stackoverflow.com/questions/79052139/numpy-random-uniform-valid-bounds-for-double
    rng = np.random.default_rng()
    expected_output = rng.uniform(min_value, max_value)
    if nptype == np.float32:
        input_data = struct.pack(">f", expected_output)
    else:
        # For float64, we use struct to pack the data
        input_data = struct.pack(">d", expected_output)

    _big_endian_test_helper(input_data, expected_output, atomic_size, format_specifier)
    _little_endian_test_helper(input_data, expected_output, atomic_size, format_specifier)


def test_unpack_str():
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

