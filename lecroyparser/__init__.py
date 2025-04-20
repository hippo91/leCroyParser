""" leCroyParser.py
(c) Benno Meier, 2018 published under an MIT license.

leCroyParser.py is derived from the matlab programme ReadLeCroyBinaryWaveform.m,
which is available at Matlab Central.
a useful resource for modifications is the LeCroy Remote Control Manual
available at http://cdn.teledynelecroy.com/files/manuals/dda-rcm-e10.pdf
------------------------------------------------------
Original version (c)2001 Hochschule fr Technik+Architektur Luzern
Fachstelle Elektronik
6048 Horw, Switzerland
Slightly modified by Alan Blankman, LeCroy Corporation, 2006

Further elements for the code were taken from pylecroy, written by Steve Bian

lecroyparser defines the ScopeData object.
Tested in Python 2.7 and Python 3.6

Updated 2020 Jeroen van Oorschot, Eindhoven University of Technology
"""

import argparse
from scope_data import ScopeData
from utils import dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse leCroy binary waveform data')
    parser.add_argument('path', type=str, help='Path to the leCroy binary waveform file')
    args = parser.parse_args()
    data = ScopeData(args.path)
    # data = ScopeData(args.path, parseAll=True)
    print(data)
    dump(data, output_filename="/tmp/toto.txt")
    dump(data)
