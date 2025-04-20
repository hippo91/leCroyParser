import sys

import numpy as np

from scope_data import ScopeData


def dump(data: ScopeData, output_filename=None):
    """
    Dump the content of the ScopeData object to the console.
    This function prints the x and y data in a formatted manner.
    It checks that the x and y data are numpy arrays of the same shape.

    :param data: ScopeData object containing the waveform data
    :return: None
    """
    assert(data.x is not None)
    assert(data.y is not None)
    assert(data.x.shape == data.y.shape)
    # Concatenate data.x and data.y into a bidimensional array
    combined_data = np.column_stack((data.x, data.y))
    if output_filename:
      np.savetxt(output_filename, combined_data, fmt="%+15.12e")
      return
    np.savetxt(sys.stdout, combined_data, fmt="%+15.12e")