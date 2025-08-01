from hashlib import file_digest
from pathlib import Path
from random import choice

from lecroyparser.scope_data_func import convert_to_text_file


def test_convert_to_text_file(tmp_path):
    """Test conversion of LeCroy binary waveform files to text files."""
    # Use Path(__file__) to get the directory of the current test file
    test_dir = Path(__file__).parent
    data_dir = test_dir / "data"
    
    # Explicit list of test files - intentionally curated for testing
    expected_test_files = [
        "Z1Trace00055.trc", 
        "Z1Trace00123.trc", 
        "Z1Trace00248.trc",
        "Z1Trace00389.trc", 
        "Z1Trace00416.trc"
    ]
    
    # Verify all expected files exist and filter to only available ones
    available_files = [
        filename for filename in expected_test_files 
        if (data_dir / filename).exists()
    ]
    assert available_files, f"No test files found. Expected: {expected_test_files}"
    
    selected_file = choice(available_files)
    file_path = data_dir / selected_file
    
    # Use pytest's tmp_path fixture for output
    output_file = convert_to_text_file(file_path, output_dir=tmp_path)
    reference_file = data_dir / "references" / output_file.name
    
    # Verify the reference file exists
    assert reference_file.exists(), f"Reference file {reference_file} not found"
    
    # Compare the generated file with the reference file
    with output_file.open("rb") as actual:
        with reference_file.open("rb") as expected:
            print(f"Comparing {output_file} with {reference_file}")
            actual_hash = file_digest(actual, "sha256").hexdigest()
            expected_hash = file_digest(expected, "sha256").hexdigest()
            assert actual_hash == expected_hash, f"File contents don't match. Expected: {expected_hash}, Got: {actual_hash}"
