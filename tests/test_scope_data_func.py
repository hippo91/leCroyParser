# mypy: ignore-errors
from lecroyparser.parsing import unpack


def test_unpack():
    """
    Test the parse function from scope_data_func.py
    """
    # Test with a sample input
    input_data = bytes([0x07, 0xd0])  # Example byte data
    expected_output = 2000 # Expected unpacked value

    # Call the parse function
    result = unpack(data=input_data,
                   offset=0,
                   position=0,
                   length=2,
                   endianness=">",
                   format_specifier="u2")

    # Assert the result
    assert result == expected_output, f"Expected {expected_output}, but got {result}"
