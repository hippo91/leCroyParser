from pathlib import Path
from random import choice

from lecroyparser.scope_data_func import convert_to_text_file

def test_convert_to_text_file():
    data_dir = Path("tests/functional/data/")
    available_files = ["Z1Trace00055.trc", "Z1Trace00123.trc", "Z1Trace00248.trc",
                       "Z1Trace00389.trc", "Z1Trace00416.trc"]
    selected_file = choice(available_files)
    file_path = data_dir / selected_file
    convert_to_text_file(file_path.as_posix())