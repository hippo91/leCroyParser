"""
lecroyparser.scope_data_fun

Reimplements initial ScopeData class from the original lecroyparser
but with a functional approach.
Introduces type hints and uses numpy for data handling.
"""

from functools import partial
import sys
from typing import Any, Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lecroyparser.metadata import MetaData
from lecroyparser.parsing import parse_data


def compose(f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose two functions.

    :param f: The first function
    :param g: The second function
    :return: The composed function
    """
    return lambda x: f(g(x))


def find_channels_files(first_channel_file_path: str) -> list[str]:
    """
    Find all leCroy binary waveform files in the given directory.

    :param first_channel_file_path: Path to the first channel file
    :return: List of paths to the leCroy binary waveform files
    """
    filepath = Path(first_channel_file_path)
    dir_path = filepath.parent
    filename = filepath.name
    if not filename.endswith(".trc"):
        raise ValueError("The file must be a .trc file")
    if not filename.startswith("C"):
        raise ValueError("The file must start with C")
    if not dir_path.is_dir():
        raise ValueError("The path must be a directory")
    channel_common_part = filename[2:]
    return [str(_path) for _path in dir_path.glob(f"C[0-9]{channel_common_part}")]


def parse_data_from_multiple_files(
    filenames: list[str], sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData | None]:
    """
    Parse the data from multiple leCroy binary waveform files.

    :param filenames: List of paths to the leCroy binary waveform files
    :param sparse: Number of points to skip in the x and y data
    :param secondDigits: Number of digits after the decimal point for seconds
    :return: Tuple of data and metadata
    """
    data: npt.NDArray[np.floating[Any]] = np.ndarray([])
    meta = None
    for filename in filenames:
        assert filename.endswith(".trc")
        data_temp, meta_temp = parse_data_from_file(
            filename, sparse=sparse, second_digits=second_digits
        )
        data = np.column_stack((data, data_temp[:, 1])) if data.size else data_temp
        if meta is None:
            meta = meta_temp
    return data, meta


def parse_data_from_file(
    filename: str, sparse: int = -1, second_digits: int = 3
) -> tuple[npt.NDArray[np.floating[Any]], MetaData]:
    """
    Parse the data from a leCroy binary waveform file.

    :param filename: Path to the leCroy binary waveform file
    :param sparse: Number of points to skip in the x and y data
    :param secondDigits: Number of digits after the decimal point for seconds
    :return: Tuple of data and metadata
    """
    assert filename.endswith(".trc")
    with open(filename, "rb") as f:
        content = f.read()
    return parse_data(content, sparse=sparse, second_digits=second_digits)


def dump(
    data: npt.NDArray[np.floating[Any]],
    metadata: Optional[MetaData] = None,
    output_filename: Optional[str] = None,
) -> None:
    """
    Dump the content of the data object to the console or to the file in argument if any.
    This function prints the x and y data in a formatted manner.
    It checks that the x and y data are numpy arrays of the same shape.

    :param data: object containing the waveform data
    :return: None
    """
    assert data.shape[1] == 2
    writer = partial(np.savetxt, X=data, header=str(metadata), fmt="%+15.12e")
    if output_filename:
        writer(output_filename)
    else:
        writer(sys.stdout)


def convert_to_text_file(
    filename: str, output_dir: str = None, sparse: int = -1, second_digits: int = 3, parse_all: bool = False
) -> str:
    """
    Convert a leCroy binary waveform file to a text file.
    The text file will contain the x and y data in a formatted manner.
    The output file will have the same name as the input file, but with a .dat extension.

    :param filename: Path to the leCroy binary waveform file
    :return: Path to the output text file
    """
    if parse_all:
        filenames = find_channels_files(filename)
        data, meta = parse_data_from_multiple_files(
            filenames, sparse=sparse, second_digits=second_digits
        )
    else:
        data, meta = parse_data_from_file(
            filename, sparse=sparse, second_digits=second_digits
        )
    output_name = filename.replace(".trc", ".dat")
    if output_dir:
        output_name = (Path(output_dir) / Path(output_name).name).as_posix()
    dump(data, metadata=meta, output_filename=output_name)
    return output_name
