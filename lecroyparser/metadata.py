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

from collections import namedtuple

MetaData = namedtuple(
    "MetaData",
    [
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
    ],
)
