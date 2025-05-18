"""
Test the unpack function from parsing.py
"""
import struct
# mypy: ignore-errors
from lecroyparser.parsing import unpack


def test_unpack_null_offset_null_position():
    """
    Test the unpack function from scope_data_func.py

    Offset and position are both 0
    The input data is a 2-byte unsigned integer
    The expected output is the integer value of the input data

    The test checks both big-endian and little-endian formats
    """
    offset = 0
    position = 0

    expected_output = 2000
    input_data = expected_output.to_bytes(2, byteorder="big")

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


def test_unpack_null_offset_null_position_float():
    """
    Test the unpack function from scope_data_func.py

    Offset and position are both 0
    The input data is a 4-byte float
    The expected output is the float value of the input data

    The test checks both big-endian and little-endian formats
    """
    offset = 0
    position = 0

    expected_output = 2000.0
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