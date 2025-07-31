"""
LeCroy Parser
This module provides the MetaData class, which is used to store metadata information
from LeCroy oscilloscopes. The class is a named tuple that contains various attributes
related to the oscilloscope data, such as template name, communication type, wave descriptor,
user text, trigger time array, wave array, instrument name, instrument number, trace label,
wave array count, vertical gain, vertical offset, nominal bits, horizontal interval,
horizontal offset, vertical unit, horizontal unit, sequence segments, trigger time,
record type, processing done, time base, vertical coupling, bandwidth limit, and wave source.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MetaData:  # pylint: disable=too-many-instance-attributes
    """Class to hold metadata information from LeCroy oscilloscopes.
    Attributes:
        template_name (str): The name of the template.
        comm_type (int): The communication type.
        wave_descriptor (int): The wave descriptor.
        user_text (int): The user text.
        trig_time_array (int): The trigger time array.
        wave_array1 (int): The wave array 1.
        instrument_name (str): The name of the instrument.
        instrument_number (int): The number of the instrument.
        trace_label (str): The label of the trace.
        wave_array_count (np.int32): The count of the wave array.
        vertical_gain (np.float32): The vertical gain.
        vertical_offset (np.float32): The vertical offset.
        nominal_bits (int): The nominal bits.
        horiz_interval (np.float32): The horizontal interval.
        horiz_offset (np.float64): The horizontal offset.
        vert_unit (str): The vertical unit.
        hor_unit (str): The horizontal unit.
        sequence_segments (int): The number of sequence segments.
        trigger_time (str): The trigger time.
        record_type (str): The type of the record.
        processing_done (str): The processing done.
        time_base (str): The time base.
        vertical_coupling (str): The vertical coupling.
        bandwidth_limit (str): The bandwidth limit.
        wave_source (str): The source of the wave.
    """
    template_name: str
    comm_type: int
    wave_descriptor: int
    user_text: int
    trig_time_array: int
    wave_array1: int
    instrument_name: str
    instrument_number: int
    trace_label: str
    wave_array_count: np.int32
    vertical_gain: np.float32
    vertical_offset: np.float32
    nominal_bits: int
    horiz_interval: np.float32
    horiz_offset: np.float64
    vert_unit: str
    hor_unit: str
    sequence_segments: int
    trigger_time: str
    record_type: str
    processing_done: str
    time_base: str
    vertical_coupling: str
    bandwidth_limit: str
    wave_source: str
