import sys

import numpy as np

from scope_data import ScopeData


def dump(data: ScopeData, output_filename: str =None):
    """
    Dump the content of the ScopeData object to the console or to the file in argument if any.
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
    else:
      np.savetxt(sys.stdout, combined_data, fmt="%+15.12e")


def convert_to_text_file(filename: str):
    """
    Convert a leCroy binary waveform file to a text file.
    The text file will contain the x and y data in a formatted manner.
    The output file will have the same name as the input file, but with a .dat extension.

    :param filename: Path to the leCroy binary waveform file
    :return: None
    """
    assert(filename.endswith(".trc"))
    data = ScopeData(filename)
    output_name = filename.replace(".trc", ".dat")
    dump(data, output_filename=output_name)