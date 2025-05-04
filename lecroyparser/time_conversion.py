"""
LeCroy Parser - Time Conversion Module
This module provides functions to convert time stamps and time base numbers
from LeCroy oscilloscopes into human-readable formats.
It includes functions to convert time stamps into a formatted string
and to convert time base numbers into a human-readable format.
"""


def convert_time_stamp(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    seconds: float,
    minutes: int,
    hours: int,
    days: int,
    months: int,
    years: int,
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
        return f"{value} " + unit.strip() + "s/div"
    if time_base_number == 100:
        return "EXTERNAL"
    raise ValueError(f"Invalid time base number: {time_base_number}")
